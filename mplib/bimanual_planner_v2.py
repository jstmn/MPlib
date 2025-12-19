from __future__ import annotations
import os
import numpy as np
import copy
from pathlib import Path
from mplib.planner import Planner 
from mplib.pymp import Pose
import pinocchio
class BimanualPlanner_v2:
    def __init__(self, urdf_path, user_link_names, srdf_path=None):
        """
        Version-2 Bimanual Planner specialized for 'dual_panda_table.urdf'
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found at: {urdf_path}")
        
        self.urdf_path = urdf_path
        self.user_link_names = user_link_names
        
        # 1. Update Joint Names according to dual_panda_table.urdf
        self.left_joint_names = [f"panda_1_joint{i}" for i in range(1, 8)]
        self.right_joint_names = [f"panda_2_joint{i}" for i in range(1, 8)]
        
        # 2. Generate/Load Dual-Arm SRDF
        if srdf_path is None:
            srdf_path = str(Path(urdf_path).with_suffix("")) + "_generated_dual.srdf"
            self._generate_dual_arm_srdf(srdf_path, self.left_joint_names, self.right_joint_names)
            
        # 3. Initialize MPLIB with the "dual_arm" group
        self.planner = Planner(
            urdf_fname=urdf_path,
            srdf_fname=srdf_path,
            move_group="dual_arm",  
            user_link_names=user_link_names,
            user_joint_names=self.left_joint_names + self.right_joint_names, 
            verbose=False
        )

        # 4. Cache Indices for IK Masking
        full_joint_names = self.planner.get_user_joint_names()
        self.left_arm_indices = [i for i, n in enumerate(full_joint_names) if n in self.left_joint_names]
        self.right_arm_indices = [i for i, n in enumerate(full_joint_names) if n in self.right_joint_names]
        print(self.left_arm_indices)
        # 5. Define End Effectors
        self.left_ee_frame = "panda_1_hand_tcp" 
        self.right_ee_frame = "panda_2_hand_tcp" 
        self.left_ee_idx = self.planner.link_name_2_idx[self.left_ee_frame]
        self.right_ee_idx = self.planner.link_name_2_idx[self.right_ee_frame]
        
        # 6. Grasp State (Stores relative transforms when holding object)
        self.grasp_T_L = None # Transform from Left EE to Object
        self.grasp_T_R = None # Transform from Right EE to Object

        print(f"Bimanual Planner V2 Initialized for {urdf_path}")

    def _generate_dual_arm_srdf(self, save_path, left_joints, right_joints):
        srdf_content = f"""<?xml version="1.0" ?>
<robot name="dual_panda_setup">
    <group name="dual_arm">
        {''.join([f'<joint name="{j}" />' for j in left_joints + right_joints])}
    </group>
    <disable_collisions link1="table" link2="panda_1_link1" reason="Adjacent" />
    <disable_collisions link1="table" link2="panda_2_link1" reason="Adjacent" />
    <disable_collisions link1="panda_1_link1" link2="panda_2_link1" reason="Never" />
</robot>"""
        with open(save_path, "w") as f:
            f.write(srdf_content)

    # --- EXISTING: Sequential IK + RRT (Your requested 'plan_bimanual') ---
    def get_ik(self, target_pose_left=None, target_pose_right=None, current_q=None):
        if current_q is None: current_q = self.planner.robot.get_qpos()
        dof = len(current_q)
        mask_left, mask_right = np.zeros(dof), np.zeros(dof)
        mask_left[self.left_arm_indices] = 1
        mask_right[self.right_arm_indices] = 1

        if target_pose_left is not None:
            status_l, q_step_1 = self._solve_single_ik(target_pose_left, current_q, self.left_ee_idx, mask_left)
            if status_l != "Success": return "Fail (Left IK)", current_q
        else: q_step_1 = current_q

        if target_pose_right is not None:
            status_r, q_final = self._solve_single_ik(target_pose_right, q_step_1, self.right_ee_idx, mask_right)
            if status_r != "Success": return "Fail (Right IK)", q_step_1
        else: q_final = q_step_1
        return "Success", q_final

    def _solve_single_ik(self, pose, start_q, link_idx, mask):
        result_q, success = self.planner.IK(goal_pose=pose, start_qpos=start_q, mask=mask)
        return ("Success", result_q) if success == "Success" else ("Fail", start_q)

    def plan_bimanual(self, target_pose_left, target_pose_right, current_q=None):
        """Standard RRT planning (independent of object constraints)"""
        if current_q is None: current_q = self.planner.robot.get_qpos()
        status, goal_q = self.get_ik(target_pose_left, target_pose_right, current_q)
        if status != "Success": return None
        return self.planner.plan_qpos(goal_qposes=[goal_q], current_qpos=current_q, planning_time=2.0, rrt_range=0.1)

    # --- HELPER: Capture Grasp ---
    def set_grasp_transforms(self, current_obj_pose, current_q=None):
        """
        Call this ONCE when the robot grasps the object.
        Calculates: T_hand_to_obj = inv(T_world_to_hand) * T_world_to_obj
        """
        if current_q is None: current_q = self.planner.robot.get_qpos()
        self.planner.robot.set_qpos(current_q, True) # Update FK
        
        # Get Current EE Poses
        pose_L = self.planner.pinocchio_model.get_link_pose(self.left_ee_idx)
        pose_R = self.planner.pinocchio_model.get_link_pose(self.right_ee_idx)
        
        # Calculate Relative Transforms (Grasp Constraints)
        # T_L_obj = T_world_L^-1 * T_world_obj
        self.grasp_T_L = pose_L.inv() * current_obj_pose
        self.grasp_T_R = pose_R.inv() * current_obj_pose
        
        print("[INFO] Grasp transforms captured.")

    # --- CORE: Plan Bimanual Screw (Jacobian Based) ---
    def plan_bimanual_screw(self, target_obj_pose, current_q=None, qpos_step=0.1, time_step=0.1):
        """
        Moves the object to 'target_obj_pose' using a linear/screw motion.
        Drives BOTH arms simultaneously to maintain the grasp.
        """
        if self.grasp_T_L is None or self.grasp_T_R is None:
            print("[ERROR] Grasp transforms not set! Call set_grasp_transforms() first.")
            return None
            
        if current_q is None: current_q = self.planner.robot.get_qpos()
        
        # 1. Calculate Goal Poses for Arms based on Object Target
        # T_world_L_goal = T_world_obj_goal * T_L_obj^-1
        target_pose_L = target_obj_pose * self.grasp_T_L.inv()
        target_pose_R = target_obj_pose * self.grasp_T_R.inv()
        
        # 2. Iterative Jacobian Descent (Dual Arm)
        # We manually implement the loop to ensure synchronized stepping
        path = [current_q.copy()]
        curr_q = current_q.copy()
        
        # Internal helper from planner.py logic
        def pose2exp(pose):
            """Computes twist (omega*theta) between identity and pose"""
            mat = pose.to_transformation_matrix()
            # Simple rotation extraction for small steps (or use pinocchio log)
            # Using MPlib's internal logic replica for simplicity
            try:
                # Calculate relative transform from current to goal
                return pinocchio.log6(pinocchio.SE3(mat)).vector
            except:
                return np.zeros(6)

        # Loop until convergence or failure
        for _ in range(1000): # Max iterations
            self.planner.robot.set_qpos(curr_q, True) # Update Forward Kinematics
            
            # A. Get Current Arm Poses
            curr_pose_L = self.planner.pinocchio_model.get_link_pose(self.left_ee_idx)
            curr_pose_R = self.planner.pinocchio_model.get_link_pose(self.right_ee_idx)
            
            # B. Calculate Twists (Error) to Targets
            # Error = Goal * Current^-1
            err_L = target_pose_L * curr_pose_L.inv()
            err_R = target_pose_R * curr_pose_R.inv()
            
            # Convert SE3 error to twist vector (v, w)
            # Note: We use pinocchio for robust math if available, else manual
            twist_L = pinocchio.log6(pinocchio.SE3(err_L.to_transformation_matrix())).vector
            twist_R = pinocchio.log6(pinocchio.SE3(err_R.to_transformation_matrix())).vector
            
            # Check Convergence
            norm = np.linalg.norm(twist_L) + np.linalg.norm(twist_R)
            if norm < 1e-3:
                print("[SUCCESS] Screw motion converged.")
                break
                
            # C. Calculate Jacobians
            J_L = self.planner.pinocchio_model.get_link_jacobian(self.left_ee_idx, local=False)
            J_R = self.planner.pinocchio_model.get_link_jacobian(self.right_ee_idx, local=False)
            
            # D. Solve Delta Q (Independently but simultaneously)
            # Since arms are kinematically independent chains (connected at base), 
            # we can solve J*dq = v for each side.
            
            # Damping factor for pseudo-inverse
            damp = 1e-4
            
            # Left Update
            # J_L is 6x14. We mask out Right Arm columns to be safe/clean
            J_L_active = J_L[:, self.left_arm_indices]
            dq_L_active = np.linalg.pinv(J_L_active.T @ J_L_active + damp * np.eye(7)) @ J_L_active.T @ twist_L
            
            # Right Update
            J_R_active = J_R[:, self.right_arm_indices]
            dq_R_active = np.linalg.pinv(J_R_active.T @ J_R_active + damp * np.eye(7)) @ J_R_active.T @ twist_R
            
            # E. Construct Full Delta Q
            dq_total = np.zeros(len(curr_q))
            dq_total[self.left_arm_indices] = dq_L_active
            dq_total[self.right_arm_indices] = dq_R_active
            
            # Scale Step Size (Safety)
            if np.linalg.norm(dq_total) > qpos_step:
                dq_total = dq_total * (qpos_step / np.linalg.norm(dq_total))
                
            curr_q += dq_total
            
            # F. Check Validity (Collisions & Limits)
            self.planner.robot.set_qpos(curr_q, True)
            if len(self.planner.planning_world.check_collision()) > 0:
                print("[FAIL] Collision detected during screw motion.")
                return None
                
            path.append(curr_q.copy())

        # 3. Time Parameterization (TOPP)
        # reusing logic from Planner.TOPP
        try:
            times, pos, vel, acc, duration = self.planner.TOPP(np.vstack(path), time_step)
            return {
                "status": "Success",
                "time": times,
                "position": pos,
                "velocity": vel,
                "acceleration": acc,
                "duration": duration
            }
        except Exception as e:
            print(f"[FAIL] TOPP parameterization failed: {e}")
            return None

    # --- NEW: Direct OMPL Constrained Planning (Alternative to Screw) ---
    def plan_bimanual_ompl_constrained(self, target_obj_pose, current_q=None, planning_time=5.0):
        """
        Plans a collision-free path using OMPL directly (RRTConnect/Constrained),
        bypassing the standard plan_qpos. Enforces a CLOSED CHAIN constraint 
        so the arms move together holding the object.
        
        Args:
            target_obj_pose (list): [x,y,z, qw,qx,qy,qz]
            current_q (np.array): Start configuration.
            planning_time (float): Time allowed for OMPL to solve.
        """
        if self.grasp_T_L is None:
            print("[ERROR] Grasp transforms not set! Call set_grasp_transforms() first.")
            return None
            
        if current_q is None: 
            current_q = self.planner.robot.get_qpos()

        # 1. Calculate Goal Configuration (using existing IK logic)
        #    The goal must be valid and satisfy the constraint implicitly.
        status, goal_q = self.get_ik(
            target_obj_pose * self.grasp_T_L.inv(), 
            target_obj_pose * self.grasp_T_R.inv(), 
            current_q
        )
        
        if status != "Success":
            print(f"[OMPL Constrained] Failed to find valid goal IK: {status}")
            return None

        # 2. Define the Closed Chain Constraint
        #    The relative transform between Left EE and Right EE must remain constant.
        #    T_rel_fixed = inv(T_Left_start) * T_Right_start
        self.planner.robot.set_qpos(current_q, True)
        start_pose_L = self.planner.pinocchio_model.get_link_pose(self.left_ee_idx)
        start_pose_R = self.planner.pinocchio_model.get_link_pose(self.right_ee_idx)
        
        # Capture the rigid relationship required
        self.fixed_rel_T = start_pose_L.inv() * start_pose_R

        # 3. Define Callbacks for OMPL
        #    Note: 'q' passed from OMPL is just the active joints. We must pad it.
        
        def constraint_func(q_active):
            """Returns 0-vector when constraint is satisfied"""
            # Recover full q (pad active joints with existing static joints if any)
            # The planner usually handles active joints, but for safety we use the 
            # pad helper from the original planner class if accessible, or assume full q
            # if move_group is 'dual_arm' (which includes all joints).
            
            # Since our move_group 'dual_arm' has ALL joints, q_active IS q_full.
            self.planner.robot.set_qpos(q_active, True)
            
            # Get current poses
            curr_L = self.planner.pinocchio_model.get_link_pose(self.left_ee_idx)
            curr_R = self.planner.pinocchio_model.get_link_pose(self.right_ee_idx)
            
            # Calculate deviation: T_err = inv(T_fixed) * (inv(T_L) * T_R)
            curr_rel = curr_L.inv() * curr_R
            diff = self.fixed_rel_T.inv() * curr_rel
            
            # Return error vector (position + rotation vector)
            # 6-dim Error
            return pinocchio.log6(pinocchio.SE3(diff.to_transformation_matrix())).vector

        def constraint_jac(q_active):
            """Numerical Jacobian of the constraint function"""
            epsilon = 1e-4
            ndim = len(q_active)
            f0 = constraint_func(q_active)
            dim_f = len(f0)
            jac = np.zeros((dim_f, ndim))
            
            for i in range(ndim):
                q_perturbed = q_active.copy()
                q_perturbed[i] += epsilon
                f_perturbed = constraint_func(q_perturbed)
                jac[:, i] = (f_perturbed - f0) / epsilon
            return jac

        # 4. Direct Call to OMPL Backend (Bypassing self.planner.plan_qpos)
        #    We access the raw OMPLPlanner instance exposed by Mplib bindings.
        print("[INFO] Calling OMPL with Closed-Chain Constraints...")
        
        # Prepare collision checking (sanity check start)
        self.planner.robot.set_qpos(current_q, True)
        if len(self.planner.planning_world.check_collision()) > 0:
            print("[ERROR] Start state in collision!")
            return None

        # self.planner.planner is the raw mplib.planning.ompl.OMPLPlanner object
        # It takes: (start_q, goal_q, time, range, verbose, constraint_func, constraint_jac, tolerance)
        try:
            status, path = self.planner.planner.plan(
                start_state=current_q,
                goal_states=[goal_q],
                time=planning_time,
                range=0.05, # Smaller step size usually better for constrained planning
                simplify=False, # Simplification often violates constraints, simpler to skip or handle carefully
                constraint_function=constraint_func,
                constraint_jacobian=constraint_jac,
                constraint_tolerance=1e-3,
                verbose=True
            )
            
            if status == "Exact solution":
                # Time Parameterize (TOPP) - Reuse existing wrapper
                times, pos, vel, acc, duration = self.planner.TOPP(path, 0.1)
                return {
                    "status": "Success",
                    "time": times,
                    "position": pos,
                    "velocity": vel,
                    "acceleration": acc,
                    "duration": duration
                }
            else:
                return {"status": f"OMPL Failed: {status}"}

        except Exception as e:
            print(f"[ERROR] OMPL Direct Call Failed: {e}")
            return None
        
        