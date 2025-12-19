#!/usr/bin/env python3
import os
import numpy as np
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import time
from mplib.bimanual_planner_v3 import BimanualPlanner_v3

class BimanualDemoSetup_v3:
    """
    Base class for bimanual motion planning demos.
    Adapted for 'dual_panda_table.urdf' (Fixed Table, No Mobile Base).
    """
    def __init__(self):
        self.sapien_to_planner_map = None
        self.planner_to_sapien_map = None

    def setup_scene(self, **kwargs):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_assets_dir = os.path.join(current_dir, "panda_assets") # Assuming URDF is in current dir
        
        # CHANGED: New URDF
        self.urdf_file = os.path.join(self.local_assets_dir, "dual_panda_table.urdf")
        self.rep_urdf_file = "/home/prajwal-vijay/Downloads/MPlib/mplib/examples/panda_assets/dual_panda_table_package_keyword_replaced.urdf"
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)

        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(kwargs.get("timestep", 1 / 240))
        self.scene.add_ground(kwargs.get("ground_height", 0))
        self.scene.default_physical_material = self.scene.create_physical_material(1, 1, 0)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        
        self.viewer = Viewer(self.renderer)
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_xyz(x=2.0, y=0.0, z=1.5)
        self.viewer.set_camera_rpy(r=0, p=-0.5, y=3.14)
        
    def debug_kinematic_tree(self):
        """Debug: Print the kinematic tree structure using Joint adjacency"""
        print("\n=== KINEMATIC TREE ===")
        
        # 1. Build Adjacency Maps (Parent -> Children) manually
        # Map: Parent Link Name -> List of Child Joints
        child_map = {}
        # Map: Child Link Name -> Parent Joint (for info printing)
        parent_joint_map = {}
        
        # Iterate over ALL joints (fixed and active) to discover topology
        for joint in self.robot.get_joints():
            parent = joint.get_parent_link()
            child = joint.get_child_link()
            
            # Store relationship if links exist
            if parent and child:
                p_name = parent.get_name()
                c_name = child.get_name()
                
                # Add to child map
                if p_name not in child_map:
                    child_map[p_name] = []
                child_map[p_name].append(joint)
                
                # Add to parent joint map
                parent_joint_map[c_name] = joint

        # 2. Recursive Print Function
        def print_link_tree(link, depth=0):
            indent = "  " * depth
            name = link.get_name()
            
            # Info about the joint connecting to this link
            joint_info = ""
            if name in parent_joint_map:
                j = parent_joint_map[name]
                joint_type = j.type  # e.g., "revolute", "fixed"
                joint_info = f" <- [Joint: {j.get_name()} ({joint_type})]"
            
            print(f"{indent}{name}{joint_info}")
            
            # Find children using the map
            if name in child_map:
                for child_joint in child_map[name]:
                    child_link = child_joint.get_child_link()
                    if child_link:
                        print_link_tree(child_link, depth + 1)
        
        # 3. Start Traversal from the Root Link
        # The first link in get_links() is always the root base
        root_link = self.robot.get_links()[0]
        print_link_tree(root_link)
        
    def load_robot(self, **kwargs):
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot: sapien.Articulation = loader.load(self.rep_urdf_file)
        # self.debug_kinematic_tree()
        # # DEBUG: Print loaded structure
        # print("\n=== ROBOT STRUCTURE ===")
        # print(f"Root pose: {self.robot.get_root_pose()}")
        # print(f"Total DOF: {self.robot.dof}")
        # print(f"Active joints: {len(self.robot.get_active_joints())}")
        
        # for i, joint in enumerate(self.robot.get_active_joints()):
        #     print(f"  Joint {i}: {joint.get_name()} - Type: {joint.type}")
        
        # print("\nLinks:")
        # for link in self.robot.get_links():
        #     pose = link.get_pose()
        #     print(f"  {link.get_name()}: pos={pose.p}, quat={pose.q}")
        print("\n=== ACTIVE JOINTS ORDER ===")
        for i, joint in enumerate(self.robot.get_active_joints()):
            print(f"{i}: {joint.get_name()}")

        # Robot is attached to world via fixed joint in URDF, so root pose is 0,0,0
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        
        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=1000, damping=200)

    def setup_planner(self, **kwargs):
        # CHANGED: Pass list of end-effectors corresponding to dual_arm group
        self.planner = BimanualPlanner_v3(
            urdf=self.urdf_file,
            srdf="/home/prajwal-vijay/Downloads/MPlib/mplib/examples/panda_assets/dual_panda_table.srdf",
            move_group="dual_arm", 
            package_keyword_replacement="",
        )
        self._create_joint_mapping()

    def _create_joint_mapping(self):
        """Create bidirectional mapping between SAPIEN and planner joint orders"""
        sapien_joint_names = [j.get_name() for j in self.active_joints]
        planner_joint_names = [self.planner.user_joint_names[idx] 
                            for idx in self.planner.move_group_joint_indices]
        
        # Map: planner_idx -> sapien_idx
        self.planner_to_sapien_map = []
        for p_name in planner_joint_names:
            if p_name in sapien_joint_names:
                self.planner_to_sapien_map.append(sapien_joint_names.index(p_name))
        
        # Map: sapien_idx -> planner_idx
        self.sapien_to_planner_map = [None] * len(sapien_joint_names)
        for planner_idx, sapien_idx in enumerate(self.planner_to_sapien_map):
            self.sapien_to_planner_map[sapien_idx] = planner_idx
    
    def follow_path(self, result):
        positions = result["position"]
        velocities = result.get("velocity", None)
        n_step = positions.shape[0]
        
        # Map planner joint names to Sapien joints
        name_to_sapien_joint = {j.get_name(): j for j in self.active_joints}
        planner_joint_names = [self.planner.user_joint_names[idx] for idx in self.planner.move_group_joint_indices]

        for i in range(n_step):
            qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            
        for planner_idx, sapien_idx in enumerate(self.planner_to_sapien_map):
            if sapien_idx < len(self.active_joints):
                sapien_joint = self.active_joints[sapien_idx]
                sapien_joint.set_drive_target(positions[i][planner_idx])
                if velocities is not None:
                    sapien_joint.set_drive_velocity_target(velocities[i][planner_idx])
                else:
                    sapien_joint.set_drive_velocity_target(0)

            
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()
            
            # If no velocity profile (e.g. raw IK path), slow down
            if velocities is None:
                time.sleep(0.02)

    def set_gripper(self, hand, pos):
        """
        Gripper Mapping for dual_panda_table.urdf:
        - Panda 1 (Right side y=-0.6): Joints 7, 8
        - Panda 2 (Left side y=+0.6): Joints 16, 17
        (Based on 7 DOF arm + 2 DOF gripper sequence)
        """
        if hand == "panda_1" or hand == "right":
            finger_indices = [14, 15]  # Updated indices
        elif hand == "panda_2" or hand == "left":
            finger_indices = [16, 17]  # Updated indices
        else:
            raise ValueError("hand must be 'panda_1'/'right' or 'panda_2'/'left'")

        for joint_idx in finger_indices:
            if joint_idx < len(self.active_joints):
                self.active_joints[joint_idx].set_drive_target(pos)
        
        for _ in range(20):
            self.scene.step()
            self.scene.update_render()
            self.viewer.render()


    def open_left_gripper(self): self.set_gripper("left", 0.04)
    def close_left_gripper(self): self.set_gripper("left", 0.0)
    def open_right_gripper(self): self.set_gripper("right", 0.04)
    def close_right_gripper(self): self.set_gripper("right", 0.0)
    def open_both_grippers(self): 
        self.open_left_gripper()
        self.open_right_gripper()
    def close_both_grippers(self):
        self.close_left_gripper()
        self.close_right_gripper()

    def move_to_pose_pair_with_RRTConnect(self, left_pose, right_pose, use_point_cloud=False, use_attach=False):
        # Get current qpos in SAPIEN ordering
        current_qpos_sapien = self.robot.get_qpos()
        
        # Convert to planner ordering (only move_group joints)
        current_qpos_planner = np.zeros(len(self.planner.move_group_joint_indices))
        for planner_idx, sapien_idx in enumerate(self.planner_to_sapien_map):
            if sapien_idx < len(current_qpos_sapien):
                current_qpos_planner[planner_idx] = current_qpos_sapien[sapien_idx]
        
        ik_result = self.planner.IK(
            left_target_pose=left_pose,
            right_target_pose=right_pose,
            start_qpos=current_qpos_planner,  # Planner ordering
            left_link_name="panda_2_hand_tcp", 
            right_link_name="panda_1_hand_tcp",
        )
        
        if isinstance(ik_result, str):
            print(f"IK Failed: {ik_result}")
            return -1
        
        collisions1 = self.planner.check_for_self_collision(qpos=ik_result)
        collisions2 = self.planner.check_for_env_collision(qpos=ik_result)
        
        if len(collisions1) > 0 or len(collisions2) > 0:
            print(f"❌ IK collision detected")
            # Print Self-Collisions
            for c in collisions1:
                print(f"   Self Collision: {c.link_name1} <--> {c.link_name2}")
            
            # Print Environment Collisions (Robot hitting table/boxes)
            for c in collisions2:
                print(f"   Env Collision: {c.link_name1} <--> {c.link_name2}")
                
            return -1
        
        goal_qposes = [ik_result]
        result = self.planner.plan_qpos_to_qpos(
            goal_qposes=goal_qposes,
            current_qpos=current_qpos_planner,  # Planner ordering
            time_step=1 / 250,
            planning_time=10.0,
            planner_name="RRTConnect",
        )

        if result["status"] != "Success":
            print(f"Planning failed: {result['status']}")
            return -1

        self.follow_path(result)
        return 0

    def move_to_pose_pair(self, left_pose, right_pose, with_screw=False):        
        return self.move_to_pose_pair_with_RRTConnect(left_pose, right_pose)