#!/usr/bin/env python3
import os
import numpy as np
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import time
from mplib.bimanual_planner import BimanualPlanner


class BimanualDemoSetup:
    """
    Base class for bimanual motion planning demos.
    Extends the single-arm DemoSetup to support dual-arm robots.
    You will need to install Sapien via `pip install sapien` for this to work
    if you want to use the viewer.
    """
    def __init__(self):
        """Initialize bimanual demo setup"""
        pass

    def setup_scene(self, **kwargs):
        """
        Setup the Sapien simulator scene.
        This is independent of mplib and specific to Sapien physics simulation.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_assets_dir = os.path.join(current_dir, "panda_assets")

        self.urdf_file = os.path.join(self.local_assets_dir, "mobile_panda_dual_arm.urdf")

        # declare sapien sim
        self.engine = sapien.Engine()
        # declare sapien renderer
        self.renderer = sapien.SapienRenderer()
        # give renderer to sapien sim
        self.engine.set_renderer(self.renderer)

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        # set simulation timestep
        self.scene.set_timestep(kwargs.get("timestep", 1 / 240))
        # add ground to scene
        self.scene.add_ground(kwargs.get("ground_height", 0))
        # set default physical material
        self.scene.default_physical_material = self.scene.create_physical_material(
            kwargs.get("static_friction", 1),
            kwargs.get("dynamic_friction", 1),
            kwargs.get("restitution", 0),
        )
        # give some white ambient light of moderate intensity
        self.scene.set_ambient_light(kwargs.get("ambient_light", [0.5, 0.5, 0.5]))
        # default enable shadow unless specified otherwise
        shadow = kwargs.get("shadow", True)
        # default spotlight angle and intensity
        direction_lights = kwargs.get(
            "direction_lights", [[[0, 1, -1], [0.5, 0.5, 0.5]]]
        )
        for direction_light in direction_lights:
            self.scene.add_directional_light(
                direction_light[0], direction_light[1], shadow=shadow
            )
        # default point lights position and intensity
        point_lights = kwargs.get(
            "point_lights",
            [[[1, 2, 2], [1, 1, 1]], [[1, -2, 2], [1, 1, 1]], [[-1, 0, 1], [1, 1, 1]]],
        )
        for point_light in point_lights:
            self.scene.add_point_light(point_light[0], point_light[1], shadow=shadow)

        # initialize viewer with camera position and orientation
        self.viewer = Viewer(self.renderer)
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_xyz(
            x=kwargs.get("camera_xyz_x", 1.2),
            y=kwargs.get("camera_xyz_y", 0.25),
            z=kwargs.get("camera_xyz_z", 0.8),
        )
        self.viewer.set_camera_rpy(
            r=kwargs.get("camera_rpy_r", 0),
            p=kwargs.get("camera_rpy_p", -0.3),
            y=kwargs.get("camera_rpy_y", 2.7),
        )

    def load_robot(self, **kwargs):
        """
        Load a bimanual robot from URDF into the Sapien scene.
        setup_scene() must be called before this function.
        """
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        
        self.robot: sapien.Articulation = loader.load(
            kwargs.get("urdf_path", self.urdf_file)
        )
        
        self.robot.set_root_pose(
            sapien.Pose(
                # FIX: Lift robot by 1cm (0.01) to clear the floor
                kwargs.get("robot_origin_xyz", [0, 0, 0.0]), 
                kwargs.get("robot_origin_quat", [1, 0, 0, 0]),
            )
        )
    
        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(
                stiffness=kwargs.get("joint_stiffness", 1000),
                damping=kwargs.get("joint_damping", 200),
            )

    def setup_planner(self, **kwargs):
        """
        Create a BimanualPlanner for dual-arm motion planning.
        
        Args:
            urdf_path: Path to the dual-arm URDF file
            srdf_path: Path to the SRDF file (optional)
            move_group: Can be "dual_arm" or ["left_panda_hand", "right_panda_hand"]
        """
        self.planner = BimanualPlanner(
            # urdf=kwargs.get("urdf_path", "/home/prajwal-vijay/Downloads/ManiSkill-main/maniskill_env/lib/python3.12/site-packages/mplib/panda/mobile_panda_dual_arm.urdf"),
            # urdf=kwargs.get("urdf_path", os.path.join(local_panda_dir, "mobile_panda_dual_arm.urdf")),
            # srdf=kwargs.get("srdf_path", ""),
            # move_group=kwargs.get("move_group", "dual_arm"),
            # include_mobile_base=kwargs.get("include_mobile_base", False),
            # package_keyword_replacement=""
            urdf=self.urdf_file,
            srdf=kwargs.get("srdf_path", ""),
            # move_group=kwargs.get("move_group", "dual_arm"),
            move_group="dual_arm",
            include_mobile_base=kwargs.get("include_mobile_base", False),

            # KEY FIX: Point package replacement to the local writable folder
            package_keyword_replacement= "/"
        )

    # def follow_path(self, result):
    #     """Helper function to follow a bimanual path generated by the planner"""
    #     # number of waypoints in the path
    #     n_step = result["position"].shape[0]
    #     # follow the path
    #     for i in range(n_step):
    #         qf = self.robot.compute_passive_force(
    #             # external=False, 
    #             gravity=True, 
    #             coriolis_and_centrifugal=True
    #         )
    #         self.robot.set_qf(qf)
    #         # set the joint positions and velocities for move group joints only
    #         for j in range(len(self.planner.move_group_joint_indices)):
    #             joint_idx = self.planner.move_group_joint_indices[j]
    #             self.active_joints[joint_idx].set_drive_target(result["position"][i][j])
    #             if hasattr(result, "velocity"):
    #                 self.active_joints[joint_idx].set_drive_velocity_target(
    #                     result["velocity"][i][j]
    #                 )
    #             else:
    #                 self.active_joints[joint_idx].set_drive_velocity_target(0)
    #         # simulation step
    #         self.scene.step()
    #         # render every 4 simulation steps to make it faster
    #         if i % 4 == 0:
    #             self.scene.update_render()
    #             self.viewer.render()
    
    def follow_path(self, result):
        """Helper function to follow a bimanual path generated by the planner"""
        positions = result["position"]
        velocities = result.get("velocity", None) # Get velocity if it exists
        n_step = positions.shape[0]
        
        # 1. Create a robust map: Joint Name -> Sapien Joint Object
        # This prevents the "Index Mismatch" bug
        name_to_sapien_joint = {j.get_name(): j for j in self.active_joints}
        
        # 2. Get the list of names the planner is controlling
        # The planner stores these as indices, we convert them back to names
        planner_joint_names = [self.planner.user_joint_names[idx] for idx in self.planner.move_group_joint_indices]

        print(f"Executing path with {n_step} steps...")

        for i in range(n_step):
            qf = self.robot.compute_passive_force(
                # external=False, 
                gravity=True, 
                coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            
            # Loop through the joints the planner calculated
            for j, joint_name in enumerate(planner_joint_names):
                
                # Check if this planner joint actually exists in the simulator
                if joint_name in name_to_sapien_joint:
                    sapien_joint = name_to_sapien_joint[joint_name]
                    
                    # Set Position
                    target_pos = positions[i][j]
                    sapien_joint.set_drive_target(target_pos)
                    
                    # Set Velocity (if TOPP succeeded)
                    if velocities is not None:
                        sapien_joint.set_drive_velocity_target(velocities[i][j])
                    else:
                        sapien_joint.set_drive_velocity_target(0)
            
            self.scene.step()
            
            # 3. Visualization Handling
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()
            
            # 4. SAFETY SLOWDOWN (If path is raw/short)
            # If TOPP failed, we don't have timestamps, so the loop runs super fast.
            # We assume a fixed dt (e.g. 0.05s) to make it visible.
            if velocities is None:
                time.sleep(0.02)
    
    def lock_base_joints(self):
        """
        Locks the mobile base joints to position 0.0 with high stiffness
        so they resist collision forces.
        """
        # Indices of base joints in your specific URDF
        # (root_x, root_y, root_z_rot, linear_actuator)
        base_indices = [0, 1, 2, 3] 
        
        for idx in base_indices:
            # 1. Set the target position to 0.0 (Hold Zero)
            # if idx != 3:
            self.active_joints[idx].set_drive_target(0.0)
            # else:
            #     self.active_joints[idx].set_drive_target(1.55)
            # 2. Crank up the stiffness (The "Holding Power")
            # 10,000 is very stiff, like a strong brake
            self.active_joints[idx].set_drive_property(stiffness=100000000, damping=100000)
    
    def set_gripper(self, hand, pos):
        """
        Helper function to activate gripper joints
        Args:
            hand: "left" or "right"
            pos: position of the gripper joint in real number (0.0 = closed, 0.4 = open)
        """
        # For Panda grippers, the last two joints are the finger joints
        # Left arm finger indices: typically 7, 8
        # Right arm finger indices: typically 15, 16
        if hand == "left":
            finger_indices = [7, 8]  # Adjust based on your URDF joint ordering
        elif hand == "right":
            finger_indices = [15, 16]  # Adjust based on your URDF joint ordering
        else:
            raise ValueError("hand must be 'left' or 'right'")

        for joint_idx in finger_indices:
            self.active_joints[joint_idx].set_drive_target(pos)

        # 100 steps is plenty to reach the target position
        for i in range(100):
            qf = self.robot.compute_passive_force(
                # external=False,
                gravity=True, 
                coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def open_left_gripper(self):
        """Open left gripper"""
        self.set_gripper("left", 0.4)

    def close_left_gripper(self):
        """Close left gripper"""
        self.set_gripper("left", 0.0)

    def open_right_gripper(self):
        """Open right gripper"""
        self.set_gripper("right", 0.4)

    def close_right_gripper(self):
        """Close right gripper"""
        self.set_gripper("right", 0.0)

    def open_both_grippers(self):
        """Open both grippers"""
        self.open_left_gripper()
        self.open_right_gripper()

    def close_both_grippers(self):
        """Close both grippers"""
        self.close_left_gripper()
        self.close_right_gripper()

    def move_to_pose_pair_with_RRTConnect(
        self,
        left_pose,
        right_pose,
        use_point_cloud=False,
        use_attach=False,
    ):
        """
        Plan and follow a path to a pair of poses using RRTConnect

        Args:
            left_pose: [x, y, z, qx, qy, qz, qw] for left end-effector
            right_pose: [x, y, z, qx, qy, qz, qw] for right end-effector
            use_point_cloud (optional): if to take the point cloud into consideration
                for collision checking.
            use_attach (optional): if to take the attach into consideration
                for collision checking.
        """
        # Get current robot qpos
        current_qpos = self.robot.get_qpos()
        left_pose_wxyz = [left_pose[0], left_pose[1], left_pose[2], 
                          left_pose[6], left_pose[3], left_pose[4], left_pose[5]]
        
        right_pose_wxyz = [right_pose[0], right_pose[1], right_pose[2], 
                           right_pose[6], right_pose[3], right_pose[4], right_pose[5]]
        # print(current_qpos)
        # Combine goal poses for bimanual planning
        # goal_qpos_combined = np.concatenate([
        #     np.array(left_pose),
        #     np.array(right_pose)
        # ])
                # 2. Get the CURRENT full robot state (22 DOF)
        # We need this to fill in the "gaps" (Base, Grippers)
        current_full_qpos = self.robot.get_qpos()
        ik_result = self.planner.IK(
            left_target_pose=left_pose_wxyz,
            right_target_pose=right_pose_wxyz,
            start_qpos=current_qpos,
            left_link_name="left_panda_hand",
            right_link_name="right_panda_hand",
        )
        collisions = self.planner.check_for_env_collision(qpos=ik_result)
        if len(collisions) > 0:
            print("❌ IK Solution is VALID mathematically, but INVALID physically (Collision)!")
            for c in collisions:
                 n1 = getattr(c, "link_name1", getattr(c, "object_name1", str(c)))
                 n2 = getattr(c, "link_name2", getattr(c, "object_name2", str(c)))
                 print(f"  Collision: {n1} <--> {n2}")
            return "Goal Collision"
            
        print("✓ IK Goal State is valid and collision-free.")
        goal_qposes_combined = []

        # 3. Iterate through IK solutions and build FULL 22-DOF goals
        # Note: This is a simplified matching strategy (taking the first valid pair)
        # Your actual code might have a loop to check all pairs.

        # COPY current state so we keep Base (0-3) and Grippers (11-12, 20-21) unchanged
        full_goal = current_full_qpos.copy()
        
        # Overwrite RIGHT Arm joints (Indices 4-10)
        full_goal[4:11] = right_pose
        
        # Overwrite LEFT Arm joints (Indices 13-19)
        full_goal[13:20] = left_pose
        # Add this valid 22-DOF vector to the list
        goal_qposes_combined.append(ik_result)
        
        if not goal_qposes_combined:
            print("No valid IK solution found for pose pair!")
            return "No IK Solution"
        
        result = self.planner.plan_qpos_to_qpos(
            goal_qposes=goal_qposes_combined,
            current_qpos=current_qpos,
            time_step=1 / 250,
            use_point_cloud=use_point_cloud,
            use_attach=use_attach,
            planning_time=10.0,
            planner_name="RRTConnect",
        )

        if result["status"] not in ["Success", "Success (No TOPP)"]:
            print(f"Planning failed: {result['status']}")
            return -1

        # Follow the planned path
        self.follow_path(result)
        return 0

    def move_to_pose_pair_with_screw(
        self,
        left_pose,
        right_pose,
        use_point_cloud=False,
        use_attach=False,
    ):
        """
        Interpolative planning with screw motion for both arms.
        Will not avoid collision and will fail if the path contains collision.
        """
        print("HI 8")
        # For bimanual screw motion, we use RRTConnect as fallback
        # (screw motion is for single-arm end-effector linear motion)
        return self.move_to_pose_pair_with_RRTConnect(
            left_pose, right_pose, use_point_cloud, use_attach
        )

    def move_to_pose_pair(
        self,
        left_pose,
        right_pose,
        with_screw=False,
        use_point_cloud=False,
        use_attach=False,
    ):        
        """API to plan and move both arms to target poses"""
        if with_screw:
            return self.move_to_pose_pair_with_screw(
                left_pose, right_pose, use_point_cloud, use_attach
            )
        else:
            return self.move_to_pose_pair_with_RRTConnect(
                left_pose, right_pose, use_point_cloud, use_attach
            )

    def move_dual_arm_constrained(
        self,
        target_object_pose,
        left_grasp_T,
        right_grasp_T,
        use_point_cloud=False,
    ):
        """
        Plan with closed-chain constraint (both arms holding one object).
        
        Args:
            target_object_pose: [x, y, z, qx, qy, qz, qw] target pose for the object
            left_grasp_T: 4x4 homogeneous transform for left grasp relative to object
            right_grasp_T: 4x4 homogeneous transform for right grasp relative to object
            use_point_cloud: whether to use point cloud for collision checking
        """
        current_qpos = self.robot.get_qpos()
        goal_qpos = current_qpos  # For constrained grasp, goal is implicit

        result = self.planner.plan_dual_arm_constrained(
            start_qpos=current_qpos,
            goal_qpos=goal_qpos,
            left_grasp_T=left_grasp_T,
            right_grasp_T=right_grasp_T,
            planning_time=3.0,
        )

        if result["status"] != "Success":
            print(f"Constrained planning failed: {result['status']}")
            return -1

        # Follow the planned path
        self.follow_path(result)
        return 0
