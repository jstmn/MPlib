import numpy as np
import time
import os
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from transforms3d.quaternions import quat2mat

# --- 1. SETUP & MODEL LOADING ---
def load_robot_model(urdf_path):
    # NOTE: Pinocchio needs to know where to find 'package://' meshes.
    # We assume the meshes are relative to the current folder or you have a "meshes" dir.
    # If meshes fail to load, you will see the robot but no skins, or you might need 
    # to append the directory containing 'franka_description' to package_dirs.
    package_dirs = [os.path.dirname(os.path.abspath(urdf_path))]
    
    # Load Model, Collision Model, and Visual Model
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_path, 
        package_dirs=package_dirs
    )
    
    # Create Data structures
    data = model.createData()
    return model, data, collision_model, visual_model

# --- 2. YOUR IK SOLVER (Pure Pinocchio Logic) ---
class PinocchioIKSolver:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Define Frame IDs for End Effectors
        # Using TCP frames from your URDF
        self.left_ee_id = model.getFrameId("panda_2_hand_tcp")
        self.right_ee_id = model.getFrameId("panda_1_hand_tcp")
        
        # Identify Active Joints for Arms (filtering out statics)
        # In Pinocchio, names often match URDF joint names
        self.active_indices = []
        for i, name in enumerate(model.names):
            # Simple filter: Grab panda joints that aren't fingers
            if "panda" in name and "finger" not in name and "hand" not in name:
                # Pinocchio joint indices might differ from q indices, 
                # but for single-DOF joints in a serial chain, we map via idx_q
                j_id = model.getJointId(name)
                idx_q = model.joints[j_id].idx_q
                if idx_q >= 0:
                    self.active_indices.append(idx_q)
        
        # Sort indices to ensure vector order
        self.active_indices = sorted(list(set(self.active_indices)))
        self.joint_limits_min = model.lowerPositionLimit
        self.joint_limits_max = model.upperPositionLimit

    def IK(self, left_target_pose=None, right_target_pose=None, start_q=None, 
           threshold=1e-3, max_iter=100, step_size=0.1):
        
        if start_q is None:
            start_q = pin.neutral(self.model)
            
        q = start_q.copy()
        
        # Helper: list [x,y,z,qw,qx,qy,qz] -> pin.SE3
        def to_SE3(pose_7d):
            if pose_7d is None: return None
            pos = np.array(pose_7d[:3])
            # URDF/Mplib is [qw, qx, qy, qz] -> Pinocchio Quaternion(w, x, y, z)
            # Ensure normalization
            quat_vec = np.array(pose_7d[3:])
            quat_vec = quat_vec / np.linalg.norm(quat_vec)
            quat = pin.Quaternion(quat_vec[0], quat_vec[1], quat_vec[2], quat_vec[3])
            return pin.SE3(quat, pos)

        target_L = to_SE3(left_target_pose)
        target_R = to_SE3(right_target_pose)

        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            error_stack = []
            J_stack = []
            
            # --- LEFT ARM ---
            if target_L is not None:
                curr_L = pin.updateFramePlacement(self.model, self.data, self.left_ee_id)
                dMf = curr_L.actInv(target_L)
                err_L = pin.log(dMf).vector # 6D Twist error
                
                # Transform error to local world frame alignment if needed, 
                # but Pinocchio log(frame_inv * goal) usually gives local body error.
                # Standard CLIK: J * dq = v_local
                error_stack.append(err_L)
                
                # Get Jacobian in LOCAL frame
                J_L = pin.computeFrameJacobian(self.model, self.data, q, self.left_ee_id, pin.ReferenceFrame.LOCAL)
                # Filter columns
                J_L_filtered = J_L[:, self.active_indices]
                J_stack.append(J_L_filtered)

            # --- RIGHT ARM ---
            if target_R is not None:
                curr_R = pin.updateFramePlacement(self.model, self.data, self.right_ee_id)
                dMf = curr_R.actInv(target_R)
                err_R = pin.log(dMf).vector
                
                error_stack.append(err_R)
                
                J_R = pin.computeFrameJacobian(self.model, self.data, q, self.right_ee_id, pin.ReferenceFrame.LOCAL)
                J_R_filtered = J_R[:, self.active_indices]
                J_stack.append(J_R_filtered)

            if not error_stack: return "Failed", q
            
            err_total = np.concatenate(error_stack)
            J_total = np.vstack(J_stack)
            
            if np.linalg.norm(err_total) < threshold:
                return "Success", q
            
            # Solve Damped Least Squares: dq = J.T * inv(J*J.T + lambda*I) * err
            damp = 1e-4
            JJt = J_total @ J_total.T
            dq_reduced = J_total.T @ np.linalg.inv(JJt + damp * np.eye(len(err_total))) @ err_total
            
            # Map reduced dq back to full q
            # Note: For simple revolute joints, q_idx == v_idx. 
            # If FreeFlyer exists, logic changes. Assuming fixed base here.
            q_update = np.zeros(self.model.nv)
            q_update[self.active_indices] = dq_reduced
            
            # Integrate
            q = pin.integrate(self.model, q, q_update * step_size)
            
            # Clamp limits
            q = np.maximum(q, self.joint_limits_min)
            q = np.minimum(q, self.joint_limits_max)

        return "Failed", q

# --- 3. VISUALIZATION HELPERS ---
def viz_target(viz, name, pose_7d, color):
    # Visualize target as a small sphere + axis
    if pose_7d is None: return
    pos = pose_7d[:3]
    # Simple sphere
    import meshcat.geometry as g
    viz.viewer[name].set_object(g.Sphere(0.02), g.MeshLambertMaterial(color=color))
    viz.viewer[name].set_transform(pin.SE3(pin.Quaternion(np.array(pose_7d[3:])), np.array(pos)).homogeneous)

def run_meshcat_demo():
    urdf_path = "/home/prajwal-vijay/Downloads/MPlib/mplib/examples/panda_assets/dual_panda_table_package_keyword_replaced.urdf" # Ensure this file exists
    
    # Load Model
    print(f"Loading {urdf_path}...")
    model, data, cmodel, vmodel = load_robot_model(urdf_path)
    
    # Init Visualizer
    viz = MeshcatVisualizer(model, cmodel, vmodel)
    viz.initViewer(open=True) # Opens browser tab
    viz.loadViewerModel()
    
    # Init Solver
    solver = PinocchioIKSolver(model, data)
    
    # Initial Configuration
    q = pin.neutral(model)
    # Set generous default pose for arms
    # Map indices manually or just use neutral if defined well in URDF
    # (Panda neutral is usually vertical, which is a singularity, so let's bend them)
    # This part depends on your joint ordering.
    
    viz.display(q)
    
    print("Running IK Loop...")
    t = 0
    dt = 0.05
    
    while True:
        t += dt
        
        # --- Targets ---
        # Left (Panda 2): Circle
        # L_pos = [0.4, 0.8 + 0.15*np.cos(t), 1 + 0.15*np.sin(t)]
        L_pos = [0.3, 0.2, 1]
        L_quat = [0, 1, 0, 0] # Down
        L_target = L_pos + L_quat
        
        # Right (Panda 1): Sine Sweep
        # R_pos = [0 + 0.1*np.sin(t*1.5), -0.2, 1]
        R_pos = [0.3, -0.2, 1]
        R_quat = [0, 1, 0, 0]
        R_target = R_pos + R_quat
        
        # --- Visualize Targets ---
        viz_target(viz, "world/target_L", L_target, 0xff00ff) # Magenta
        viz_target(viz, "world/target_R", R_target, 0xffff00) # Yellow
        
        # --- Solve ---
        status, q = solver.IK(
            left_target_pose=L_target,
            right_target_pose=R_target,
            start_q=q,
            max_iter=10, # Keep iter low for real-time feel
            step_size=0.5
        )
        
        # --- Display ---
        viz.display(q)
        time.sleep(dt)

if __name__ == "__main__":
    run_meshcat_demo()