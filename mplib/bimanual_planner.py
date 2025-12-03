from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint
from transforms3d.quaternions import mat2quat, quat2mat
import pinocchio
from mplib.pymp import ArticulatedModel, PlanningWorld
from mplib.pymp.planning import ompl

class BimanualPlanner:
    def __init__(
            self,
            urdf: str,
            move_group: str | Sequence[str], # Can now accept a list of different groups that may be the target!
            srdf: str = "",
            package_keyword_replacement: str = "",
            user_link_names: Sequence[str] = [],
            user_joint_names: Sequence[str] = [],
            joint_vel_limits: Optional[Sequence[float] | np.ndarray] = None,
            joint_acc_limits: Optional[Sequence[float] | np.ndarray] = None,
            include_mobile_base: bool = False, # This flag determines if the base can move or not
            **kwargs
            ):
        if joint_vel_limits is None:
            joint_vel_limits = []
        if joint_acc_limits is None:
            joint_acc_limits = []

        self.urdf = urdf
        # ... (Standard SRDF handling logic remains here) ...
        if srdf == "" and os.path.exists(urdf.replace(".urdf", ".srdf")):
            self.srdf = urdf.replace(".urdf", ".srdf")

        # Replace package keyword logic...
        urdf = self.replace_package_keyword(package_keyword_replacement)

        # If user didn't specify joints, we load the model first to inspect it
        temp_robot = ArticulatedModel(urdf, srdf, [0, 0, -9.81], [], [], convex=True, verbose=True)
        all_joint_names = temp_robot.get_pinocchio_model().get_joint_names()
        
        # Define known base joints from your URDF to easily filter them
        base_joints = [
            "root_x_axis_joint", 
            "root_y_axis_joint", 
            "root_z_rotation_joint", 
            "linear_actuator_height"
        ]
        
        # Since our URDF is such that The base itself can move, we need to exclude it and keep the setup simple.
        if not user_joint_names:
            if include_mobile_base:
                user_joint_names = all_joint_names
            else:
                user_joint_names = [
                    jn for jn in all_joint_names 
                    if jn not in base_joints
                ]
                print(f"Excluding base joints: {base_joints}")
        
            
        self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names, # Now strictly controlled
            convex=True,
            verbose=False,
        )

        self.pinocchio_model = self.robot.get_pinocchio_model()
        self.user_link_names = self.pinocchio_model.get_link_names()
        self.user_joint_names = self.pinocchio_model.get_joint_names()

        self.planning_world = PlanningWorld(
            [self.robot],
            ["robot"],
            kwargs.get("normal_objects", []),
            kwargs.get("normal_object_names", []),
        )

        if srdf == "":
            self.generate_collision_pair() # Need to create this function
            self.robot.update_SRDF(self.srdf)

        # Map names to indices
        self.joint_name_2_idx = {j: i for i, j in enumerate(self.user_joint_names)}
        self.link_name_2_idx = {l: i for i, l in enumerate(self.user_link_names)}

        # Here we handle the move_group. We define a alias called "dual_arm"
        # According to the URDF, left_panda_hand refers to the palm of the left hand, to which
        # two fingers are attached. There is also tool center position, which is slightly in front of the left hand, and in between the two fingers.
        # It can be used for greater accuracy.
        self.move_group = move_group
        self.move_group_joint_indices = []
        target_links = []
        if move_group == "dual_arm":
            target_links = ["left_panda_hand", "right_panda_hand"]
        elif isinstance(move_group, list):
            target_links = move_group
        else:
            target_links = [move_group] # When there is only a single group. (like you want planning for only one arm, then other one is treated as a fixed obstacle.)
        
        # Validate links exist
        for link in target_links:
            assert link in self.user_link_names, f"Link {link} not found in robot model."
        # Collect joint indices for All target chains(unlike single arm, here there are multiple move_groups)
        # We donot want chains to be overlapping
        joint_indices_set = set()

        for link in target_links:
            self.robot.set_move_group(link)
            indices = self.robot.get_move_group_joint_indices()
            for idx in indices:
                # Add this joint only if it is in "user_joint_names" list
                # This is done to filter out the base joints
                if idx < len(self.user_joint_names):
                    joint_indices_set.add(idx)
        self.move_group_joint_indices = sorted(list(joint_indices_set))
        
        self.joint_types = self.pinocchio_model.get_joint_types()
        self.joint_limits = np.concatenate(self.pinocchio_model.get_joint_limits())
        
        # Standard limits handling
        self.joint_vel_limits = (
            joint_vel_limits
            if len(joint_vel_limits)
            else np.ones(len(self.move_group_joint_indices))
        )
        self.joint_acc_limits = (
            joint_acc_limits
            if len(joint_acc_limits)
            else np.ones(len(self.move_group_joint_indices))
        )
        
        # Note: move_group_link_id is tricky for dual arm. 
        # We usually just store the first one or a list.
        if len(target_links) == 1:
            self.move_group_link_id = self.link_name_2_idx[target_links[0]]
        else:
            # For dual arm, we might need a list of IDs, but let's keep specific implementation details
            # for the specific methods that need them (like IK)
            self.move_group_link_id = [self.link_name_2_idx[l] for l in target_links]

        assert len(self.joint_vel_limits) == len(self.joint_acc_limits), (
            f"length of joint_vel_limits ({len(self.joint_vel_limits)}) =/= "
            f"length of joint_acc_limits ({len(self.joint_acc_limits)})"
        )
        assert len(self.joint_vel_limits) == len(self.move_group_joint_indices), (
            f"length of joint_vel_limits ({len(self.joint_vel_limits)}) =/= "
            f"length of move_group ({len(self.move_group_joint_indices)})"
        )
        assert len(self.joint_vel_limits) <= len(self.joint_limits), (
            f"length of joint_vel_limits ({len(self.joint_vel_limits)}) > "
            f"number of total joints ({len(self.joint_limits)})"
        )

        self.planning_world = PlanningWorld([self.robot], ["robot"], [], [])
        self.planner = ompl.OMPLPlanner(world=self.planning_world)

    def replace_package_keyword(self, package_keyword_replacement):
        """
        some ROS URDF files use package:// keyword to refer the package dir
        replace it with the given string (default is empty)

        Args:
            package_keyword_replacement: the string to replace 'package://' keyword
        """
        rtn_urdf = self.urdf
        with open(self.urdf) as in_f:
            content = in_f.read()
            if "package://" in content:
                rtn_urdf = self.urdf.replace(".urdf", "_package_keyword_replaced.urdf")
                content = content.replace("package://", package_keyword_replacement)
                if not os.path.exists(rtn_urdf):
                    with open(rtn_urdf, "w") as out_f:
                        out_f.write(content)
        return rtn_urdf

    def generate_collision_pair(self, sample_time=1000000, echo_freq=100000):
        """
        We read the srdf file to get the link pairs that should not collide.
        If not provided, we need to randomly sample configurations
        to find the link pairs that will always collide.
        """
        print(
            "Since no SRDF file is provided. We will first detect link pairs that will"
            " always collide. This may take several minutes."
        )
        n_link = len(self.user_link_names)
        cnt = np.zeros((n_link, n_link), dtype=np.int32)
        for i in range(sample_time):
            qpos = self.pinocchio_model.get_random_configuration()
            self.robot.set_qpos(qpos, True)
            collisions = self.planning_world.collide_full()
            for collision in collisions:
                u = self.link_name_2_idx[collision.link_name1]
                v = self.link_name_2_idx[collision.link_name2]
                cnt[u][v] += 1
            if i % echo_freq == 0:
                print("Finish %.1f%%!" % (i * 100 / sample_time))

        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        root = ET.Element("robot")
        robot_name = self.urdf.split("/")[-1].split(".")[0]
        root.set("name", robot_name)
        self.srdf = self.urdf.replace(".urdf", ".srdf")

        for i in range(n_link):
            for j in range(n_link):
                if cnt[i][j] == sample_time:
                    link1 = self.user_link_names[i]
                    link2 = self.user_link_names[j]
                    print(
                        f"Ignore collision pair: ({link1}, {link2}), "
                        "reason: always collide"
                    )
                    collision = ET.SubElement(root, "disable_collisions")
                    collision.set("link1", link1)
                    collision.set("link2", link2)
                    collision.set("reason", "Default")
        with open(self.srdf, "w") as srdf_file:
            srdf_file.write(
                minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
            )
            srdf_file.close()
        print("Saving the SRDF file to %s" % self.srdf)

    
    def wrap_joint_limit(self, q) -> bool:
        """
        Checks if the joint configuration is within the joint limits.
        For revolute joints, the joint angle is wrapped to be within [q_min, q_min+2*pi)

        Args:
            q: joint configuration, angles of revolute joints might be modified

        Returns:
            True if q can be wrapped to be within the joint limits
        """
        n = len(q)
        flag = True
        for i in range(n):
            if self.joint_types[i].startswith("JointModelR"):
                if -1e-3 <= q[i] - self.joint_limits[i][0] < 0:
                    continue
                q[i] -= (
                    2 * np.pi * np.floor((q[i] - self.joint_limits[i][0]) / (2 * np.pi))
                )
                if q[i] > self.joint_limits[i][1] + 1e-3:
                    flag = False
            else:
                if (
                    q[i] < self.joint_limits[i][0] - 1e-3
                    or q[i] > self.joint_limits[i][1] + 1e-3
                ):
                    flag = False
        return flag

    def pad_qpos(self, qpos, articulation=None):
        """
        if the user does not provide the full qpos but only the move_group joints,
        pad the qpos with the rest of the joints
        """
        if len(qpos) == len(self.move_group_joint_indices):
            tmp = (
                articulation.get_qpos()
                if articulation is not None
                else self.robot.get_qpos()
            )
            # Iterate through the known active indices and assign them one by one
            for k, idx in enumerate(self.move_group_joint_indices):
                tmp[idx] = qpos[k]            
            qpos = tmp

        assert len(qpos) == len(self.joint_limits), (
            f"length of qpos ({len(qpos)}) =/= "
            f"number of total joints ({len(self.joint_limits)})"
        )

        return qpos

    def check_for_collision(
        self,
        collision_function,
        articulation: Optional[ArticulatedModel] = None,
        qpos: Optional[np.ndarray] = None,
    ) -> list:
        """helper function to check for collision"""
        # handle no user input
        if articulation is None:
            articulation = self.robot
        if qpos is None:
            qpos = articulation.get_qpos()
        qpos = self.pad_qpos(qpos, articulation)

        # first save the current qpos
        old_qpos = articulation.get_qpos()
        # set robot to new qpos
        articulation.set_qpos(qpos, True)
        # find the index of the articulation inside the array
        idx = self.planning_world.get_articulations().index(articulation)
        # check for self-collision
        collisions = collision_function(idx)
        # reset qpos
        articulation.set_qpos(old_qpos, True)
        return collisions
    
    def check_for_self_collision(
        self,
        articulation: Optional[ArticulatedModel] = None,
        qpos: Optional[np.ndarray] = None,
    ) -> list:
        """Check if the robot is in self-collision.

        Args:
            articulation: robot model. if none will be self.robot
            qpos: robot configuration. if none will be the current pose

        Returns:
            A list of collisions.
        """
        return self.check_for_collision(
            self.planning_world.self_collide, articulation, qpos
        )

    def check_for_env_collision(
        self,
        articulation: Optional[ArticulatedModel] = None,
        qpos: Optional[np.ndarray] = None,
        with_point_cloud=False,
        use_attach=False,
    ) -> list:
        """Check if the robot is in collision with the environment

        Args:
            articulation: robot model. if none will be self.robot
            qpos: robot configuration. if none will be the current pose
            with_point_cloud: whether to check collision against point cloud
            use_attach: whether to include the object attached to the end effector
                in collision checking
        Returns:
            A list of collisions.
        """
        # store previous results
        prev_use_point_cloud = self.planning_world.use_point_cloud
        prev_use_attach = self.planning_world.use_attach
        self.planning_world.set_use_point_cloud(with_point_cloud)
        self.planning_world.set_use_attach(use_attach)

        results = self.check_for_collision(
            self.planning_world.collide_with_others, articulation, qpos
        )

        # restore
        self.planning_world.set_use_point_cloud(prev_use_point_cloud)
        self.planning_world.set_use_attach(prev_use_attach)
        return results
    
    # This IK works only in the world frame, not the base of the bot frame
    def IK(self, left_target_pose=None, right_target_pose=None, start_qpos=None, left_link_name="left_panda_hand", right_link_name="right_panda_hand", threshold=1e-3, max_iter=100, step_size=0.1):
        """
        Inverse kinematics for bimanual robot using CLIK method.

        Args:
            left_target_pose: 4x4 homogeneous transformation matrix for left end-effector
            right_target_pose: 4x4 homogeneous transformation matrix for right end-effector
            start_qpos: initial joint configuration
            left_link_name: name of the left end-effector link
            right_link_name: name of the right end-effector link
            threshold: convergence threshold
            max_iter: maximum number of iterations
        """

        # Note we cannot use the compute_IK_CLIK directly since it only supports single end-effector.
        # We will use something that is commonly used in industry standard application to find the inverse kinematics
        # It uses gradient descent with pinnochio, computing the Jacobian for both arms and updating accordingly.

        if start_qpos is None:
            start_qpos = self.robot.get_qpos()
        
        left_idx = self.link_name_2_idx.get(left_link_name, -1)
        right_idx = self.link_name_2_idx.get(right_link_name, -1)

        # Use the active joints defined in our __init__ logic
        active_indices = self.move_group_joint_indices
        q = np.copy(start_qpos)

        # This helper function computes error in 6D, now this is different from error
        # in cartesian, this is error in position and twists(since jacobian output is in the form of angular velocity!)
        # Errors are calculated in the to_SE3 format
        # Helper: Convert 7D list [x,y,z,qw,qx,qy,qz] to pinocchio.SE3 object
        def to_SE3(pose_7d):
            # Check if pose_7d is already an SE3 object to be safe
            if isinstance(pose_7d, pinocchio.SE3): return pose_7d
            
            # Create Quaternion (w, x, y, z) -> Pinocchio uses (x, y, z, w) usually!
            # BUT: Transforms3d uses (w, x, y, z). Pinocchio Python bindings usually expect [x,y,z,w].
            # Let's rely on Rotation Matrix to be safe from Quaternion conventions.
            R = quat2mat(pose_7d[3:]) # Expects [w, x, y, z]
            t = np.array(pose_7d[:3])
            return pinocchio.SE3(R, t)
        
        target_L_se3 = to_SE3(left_target_pose)
        target_R_se3 = to_SE3(right_target_pose)

        for i in range(max_iter):
            self.pinocchio_model.compute_forward_kinematics(q)

            # Get the current pose as SE3 objects
            p_L_array = self.pinocchio_model.get_global_link_transform(left_idx)
            current_L_se3 = to_SE3(p_L_array)

            p_R_array = self.pinocchio_model.get_global_link_transform(right_idx)
            current_R_se3 = to_SE3(p_R_array)

            # Compute error using current^-1*target
            # log computes the twist(velocity) required to travel that difference

            error_L = pinocchio.log(current_L_se3.actInv(target_L_se3)).vector
            error_R = pinocchio.log(current_R_se3.actInv(target_R_se3)).vector

            error_stack = np.concatenate([error_L, error_R])  # 12D error vector

            if np.linalg.norm(error_stack) < threshold:
                print(f"IK converged in {i} iterations.")
                return q
            
            # Compute Jacobians for both end-effectors
            self.pinocchio_model.compute_full_jacobian(q) # This line will refresh and compute new matrices according to the new configurations
            
            J_L = self.pinocchio_model.get_link_jacobian(left_idx, local=False)[:, active_indices] # This will compute the jacobian for the left hand
            J_R = self.pinocchio_model.get_link_jacobian(right_idx, local=False)[:, active_indices] # This will compute the jacobian for the right hand
            # Notice above, how we filter out the base joints movement, using active indices
            # Why is this filtering valid?
            # Essentially, Jacobian says given the velocities of my joints, what is the velocity of my EE. Thats all. Each time  configuration of the arm changes
            # we might calculate new jacobian, that means a new map between the ee velocity and the joint velocities, but filtering out the columns from it, means our
            # input joint velocities corresponding to the base is 0, so the ee will not even move the base!
            # D. Solve
            J_stack = np.vstack([J_L, J_R])
            dq = np.linalg.pinv(J_stack) @ error_stack
            q[active_indices] += step_size * dq
        
        return "Failed", q
    
    # Wrapper function when an object is held
    def plan_dual_arm_grasp(
        self,
        target_object_pose,
        left_grasp_T,
        right_grasp_T,
        start_qpos,
        treshold=1e-3,
    ):
        # Covert target of object to matrix
        T_obj_world = np.eye(4)
        T_obj_world[:3, :3] = quat2mat(target_object_pose[3:])
        T_obj_world[:3, 3] = target_object_pose[:3]
        
        # Calculate Hand Targets
        T_left_target = T_obj_world @ np.linalg.inv(left_grasp_T)
        left_target_7d = np.zeros(7)
        left_target_7d[:3] = T_left_target[:3, 3]
        left_target_7d[3:] = mat2quat(T_left_target[:3, :3])
        
        T_right_target = T_obj_world @ np.linalg.inv(right_grasp_T)
        right_target_7d = np.zeros(7)
        right_target_7d[:3] = T_right_target[:3, 3]
        right_target_7d[3:] = mat2quat(T_right_target[:3, :3])
        
        # Call IK
        status, q_result = self.IK(
            left_target_7d,
            right_target_7d,
            start_qpos,
            threshold=treshold,
        )
        
        if status == "Success":
            return {
                "status": "Success",
                "qpos": q_result,
                "left_target": left_target_7d,
                "right_target": right_target_7d
            }
        else:
            return {"status": "IK Failed"}

    def create_dual_arm_constraint_function(self, left_grasp_T, right_grasp_T):
        left_idx = self.link_name_2_idx["left_panda_hand"]
        right_idx = self.link_name_2_idx["right_panda_hand"]
        
        def to_SE3(mat):
            return pinocchio.SE3(mat[:3, :3], mat[:3, 3])
        
        se3_grasp_L = to_SE3(left_grasp_T)
        se3_grasp_R = to_SE3(right_grasp_T)
        
        # What this does is, assumes that both left and right hands are holding the same object correctly.
        # Then tries to project for the object position from left and right hands independently, and finds the error.
        def constraint_function(q):
            full_q = self.pad_qpos(q)
            self.pinocchio_model.compute_forward_kinematics(full_q)
            p_L = self.pinocchio_model.get_link_pose(left_idx)
            T_L = pinocchio.SE3(quat2mat(p_L[3:]), np.array(p_L[:3]))
            
            p_R = self.pinocchio_model.get_link_pose(right_idx)
            T_R = pinocchio.SE3(quat2mat(p_R[3:]), np.array(p_R[:3]))
            
            T_Obj_L = T_L.act(se3_grasp_L)
            T_Obj_R = T_R.act(se3_grasp_R)
            
            return pinocchio.log(T_Obj_L.actInv(T_Obj_R)).vector
        
        
        def numerical_jacobian(q):
            eps = 1e-4
            n = len(q)
            J = np.zeros((6, n))
            f0 = constraint_function(q)
            
            for i in range(n):
                q_plus = q.copy() # As partial derivatives, keep it in the loop
                q_plus[i] += eps
                f_plus = constraint_function(q_plus)
                J[:, i] = (f_plus - f0) / eps # Essentially calculating the jacobian numerically. 
                # If I move by an amount eps in joint i, how much does the constraint function change?
            return J
        
        return constraint_function, numerical_jacobian
    
    def TOPP(self, path, step=0.1, verbose=False):
        """
        Time-Optimal Path Parameterization

        Args:
            path: numpy array of shape (n, dof)
            step: step size for the discretization
            verbose: if True, will print the log of TOPPRA
        """

        N_samples = path.shape[0]
        dof = path.shape[1]
        assert dof == len(self.joint_vel_limits)
        assert dof == len(self.joint_acc_limits)
        ss = np.linspace(0, 1, N_samples)
        path = ta.SplineInterpolator(ss, path)
        pc_vel = constraint.JointVelocityConstraint(self.joint_vel_limits)
        pc_acc = constraint.JointAccelerationConstraint(self.joint_acc_limits)
        instance = algo.TOPPRA(
            [pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel"
        )
        jnt_traj = instance.compute_trajectory()
        if jnt_traj is None:
            raise RuntimeError("Fail to parameterize path")
        ts_sample = np.linspace(0, jnt_traj.duration, int(jnt_traj.duration / step))
        qs_sample = jnt_traj(ts_sample)
        qds_sample = jnt_traj(ts_sample, 1)
        qdds_sample = jnt_traj(ts_sample, 2)
        return ts_sample, qs_sample, qds_sample, qdds_sample, jnt_traj.duration

    # To be called only when both arms are holding the same object!
    def plan_dual_arm_constrained(
        self,
        start_qpos,
        goal_qpos,
        left_grasp_T,
        right_grasp_T,
        time_step=0.1,
        rrt_range=0.01,
        planning_time=5,
        verbose=False):
        
        constr_func, constr_jac = self.create_dual_arm_constraint_function(left_grasp_T, right_grasp_T)
        
        return self.plan_qpos_to_qpos(
            goal_qposes=[goal_qpos],
            current_qpos=start_qpos,
            time_step=time_step,
            rrt_range=rrt_range,
            planning_time=planning_time,
            planner_name="RRTConnect",
            no_simplification=True, # Important to have valid constrained paths
            constraint_function=constr_func,
            constraint_jacobian=constr_jac,
            constraint_tolerance=1e-3,
            verbose=verbose,
        )

# Our already existing plan_qpos_to_qpos handles constrained jacobians, that is OMPL library handles it!
    def plan_qpos_to_qpos(
        self,
        goal_qposes: list,
        current_qpos,
        time_step=0.1,
        rrt_range=0.1,
        planning_time=1,
        fix_joint_limits=True,
        use_point_cloud=False,
        use_attach=False,
        planner_name="RRTConnect",
        no_simplification=False,
        constraint_function=None, # If None: Independent Arms. If Set: Constrained.
        constraint_jacobian=None,
        constraint_tolerance=1e-3,
        fixed_joint_indices=None,
        verbose=False,
    ):
        if fixed_joint_indices is None:
            fixed_joint_indices = []
        self.planning_world.set_use_point_cloud(use_point_cloud)
        self.planning_world.set_use_attach(use_attach)
        
        n = current_qpos.shape[0]
        if fix_joint_limits:
            for i in range(n):
                if current_qpos[i] < self.joint_limits[i][0]:
                    current_qpos[i] = self.joint_limits[i][0] + 1e-3
                if current_qpos[i] > self.joint_limits[i][1]:
                    current_qpos[i] = self.joint_limits[i][1] - 1e-3

        current_qpos = self.pad_qpos(current_qpos)
        
        self.robot.set_qpos(current_qpos, True)
        collisions = self.planning_world.collide_full()
        if len(collisions) != 0:
            print("Invalid start state")
            return {"status": "Invalid start state"}
        
        idx = self.move_group_joint_indices
        goal_qpos_ = [goal_qposes[i][idx] for i in range(len(goal_qposes))]
        
        fixed_joints = set()
        for joint_idx in fixed_joint_indices:
            fixed_joints.add(ompl.FixedJoint(0, joint_idx, current_qpos[joint_idx]))
            
        # OMPL Wrapper Call
        
        status, path = self.planner.plan(
            current_qpos[idx],
            goal_qpos_,
            range=rrt_range,
            time=planning_time,
            fixed_joints=fixed_joints,
            planner_name=planner_name,
            no_simplification=no_simplification,
            constraint_function=constraint_function,
            constraint_jacobian=constraint_jacobian,
            constraint_tolerance=constraint_tolerance,
            verbose=verbose,
        )
        
        if status == "Exact Solution":
            if verbose:
                ta.setup_logging("INFO")
            else:
                ta.setup_logging("WARNING")
            
            # TOPP can sometimes struggle with constrained paths if they are jagged, in that case we donot want the program to fail
            try:
                times, pos, vel, acc, duration = self.TOPP(path, time_step)
                return {
                    "status": "Success",
                    "time": times,
                    "position": pos,
                    "velocity": vel,
                    "acceleration": acc,
                    "duration": duration,
                }
            except Exception as e:
                print(f"TOPP Parameterization Failed: {e}")
                return {"status": "Success (No TOPP)", "position": path}
        else:
            return {"status": "RRT Failed. %s" % status}

    def update_point_cloud(self, pc, radius=1e-3):
        """
        Args:
            pc: numpy array of shape (n, 3)
            radius: radius of each point. This gives a buffer around each point
                that planner will avoid
        """
        self.planning_world.update_point_cloud(pc, radius)

    def update_attached_tool(self, fcl_collision_geometry, pose, link_id=-1):
        """helper function to update the attached tool"""
        if link_id == -1:
            link_id = self.move_group_link_id
        self.planning_world.update_attached_tool(fcl_collision_geometry, link_id, pose)

    def update_attached_sphere(self, radius, pose, link_id=-1):
        """
        attach a sphere to some link

        Args:
            radius: radius of the sphere
            pose: [x,y,z,qw,qx,qy,qz] pose of the sphere
            link_id: if not provided, the end effector will be the target.
        """
        if link_id == -1:
            link_id = self.move_group_link_id
        self.planning_world.update_attached_sphere(radius, link_id, pose)

    def update_attached_box(self, size, pose, link_id=-1):
        """
        attach a box to some link

        Args:
            size: [x,y,z] size of the box
            pose: [x,y,z,qw,qx,qy,qz] pose of the box
            link_id: if not provided, the end effector will be the target.
        """
        if link_id == -1:
            link_id = self.move_group_link_id
        self.planning_world.update_attached_box(size, link_id, pose)

    def update_attached_mesh(self, mesh_path, pose, link_id=-1):
        """
        attach a mesh to some link

        Args:
            mesh_path: path to the mesh
            pose: [x,y,z,qw,qx,qy,qz] pose of the mesh
            link_id: if not provided, the end effector will be the target.
        """
        if link_id == -1:
            link_id = self.move_group_link_id
        self.planning_world.update_attached_mesh(mesh_path, link_id, pose)

    def set_base_pose(self, pose):
        """
        tell the planner where the base of the robot is w.r.t the world frame

        Args:
            pose: [x,y,z,qw,qx,qy,qz] pose of the base
        """
        self.robot.set_base_pose(pose)

    def set_normal_object(self, name, collision_object):
        """adds or updates a non-articulated collision object in the scene"""
        self.planning_world.set_normal_object(name, collision_object)

    def remove_normal_object(self, name):
        """returns true if the object was removed, false if it was not found"""
        return self.planning_world.remove_normal_object(name)
