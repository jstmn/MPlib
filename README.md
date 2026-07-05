# Bimanual Motion Planning Extension

This branch contains a bimanual motion-planning extension built for coordinated
dual-arm manipulation. The implementation was developed for the Colosseum V2
project, where dual Franka Panda robots are used to generate long-horizon
demonstrations for vision-language-action model training and evaluation.

The planner extends the standard single-arm motion-planning workflow into a
dual-arm setting with synchronized IK, coordinated Cartesian motion,
collision-aware joint-space planning, and trajectory time parameterization.



[Stack 3 Cubes](https://github.com/user-attachments/assets/d0c21730-682a-44a9-a1a6-07f6ee88eccf)
[Pour Pot](https://github.com/user-attachments/assets/4d82883f-ea65-4eab-8289-646232738d83)



## Motivation

Many manipulation benchmarks focus on single-arm pick-and-place behavior. In
contrast, bimanual manipulation introduces additional planning challenges:

- Two end effectors must often satisfy task-space goals simultaneously.
- Planning must avoid self-collision between two kinematic chains sharing the
  same workspace.
- Execution needs stable conversion between planner-space joint order and the
  joint order used by a simulator or robot controller.
- Long-horizon tasks require mixing joint-space, Cartesian, and gripper
  primitives in a reusable way.

This branch addresses those issues through a custom dual-arm planner layered on
top of MPlib, OMPL, Pinocchio, and TOPPRA.

## Technical Summary

The core planner:

- Builds the planning model from a dual-arm URDF.
- Creates an MPlib `PlanningWorld` for collision-aware planning.
- Uses Pinocchio for forward kinematics, Jacobians, SE(3) errors, and
  differential IK.
- Uses OMPL/RRTConnect for joint-space planning.
- Smooths and time-parameterizes paths with shortcut smoothing and TOPPRA.
- Produces reusable trajectory outputs that can be consumed by downstream
  simulators or robot-control stacks.

This keeps the planning implementation focused on robot kinematics, collision
checking, and trajectory generation, while leaving simulator-specific execution
to downstream code.

## Core Capabilities

### Dual-Arm Robot Model Construction

The planner constructs a fixed-base dual-Panda model from URDF/SRDF assets and
extracts the full joint and link layout through Pinocchio. It then creates a
planning world with:

- The full articulated dual-arm robot.
- A `dual_arm` move group spanning both Panda arms and grippers.
- Name-to-index maps for joints and links.
- Self-collision and environment-collision queries.
- Optional normal objects, attached tools, point clouds, boxes, spheres, and
  meshes.

This setup lets the planner reason over the entire bimanual configuration
rather than treating each arm as an independent single-arm robot.

### Simultaneous Dual-Arm IK

The IK solver accepts a left target pose, a right target pose, or both. For each
active target it:

- Converts target poses into Pinocchio `SE3`.
- Computes forward kinematics for the current configuration.
- Computes local link Jacobians for the relevant TCPs.
- Forms a stacked task-space error over both arms.
- Solves a damped least-squares update:

```text
dq = J^T (J J^T + lambda I)^-1 e
```

The solver supports randomized restarts and joint-limit clipping. This allows
the system to find feasible configurations for two end-effector goals in the
same planning call.

### Collision-Aware Joint-Space Planning

For global motions, the planner calls OMPL through MPlib. The pipeline is:

1. Pad partial move-group qpos into full robot qpos.
2. Clamp the start state into joint limits.
3. Reject invalid starts with self/environment collision checks.
4. Fix gripper mimic joints during arm motion.
5. Plan to one or more goal configurations using RRTConnect.
6. Shortcut-smooth the resulting path where collision-free.
7. Densify waypoints for stable downstream execution.
8. Time-parameterize with TOPPRA under velocity/acceleration limits.

The output trajectory includes positions, velocities, accelerations, and
duration when TOPPRA succeeds, with a path-only fallback when timing fails.

### Cartesian Screw Motion

For local Cartesian moves, the planner implements differential-IK screw motion.
At each step it:

- Computes current TCP poses.
- Computes SE(3) error to each active target pose.
- Clamps the task-space error to enforce small Cartesian updates.
- Stacks the left and right arm Jacobians.
- Solves a damped least-squares joint update.
- Preserves gripper joint values.
- Optionally checks collision along the generated path.

This is useful for approach, retreat, insertion, threading, lifting, and other
motions where a straight or locally smooth end-effector path is preferred over
global RRT behavior.

## High-Level Planning Operations

The planner supports a compact set of reusable operations:

- Single-arm target-pose planning while keeping the other arm fixed.
- Dual-arm target-pose-pair planning using simultaneous IK followed by
  collision-aware joint-space planning.
- Single-arm Cartesian screw motion while holding the other arm's TCP fixed.
- Dual-arm Cartesian screw motion for simultaneous local end-effector motion.
- Trajectory generation with path smoothing, waypoint densification, and
  optional time parameterization.

These operations are designed to be composed into complete task-level solutions
outside the core planner.

## Downstream Integration

The planner was designed to be integrated with a downstream robotics simulator
or control stack for demonstration generation. The integration boundary handles
several practical issues that arise when bridging a planning library with an
execution environment:

- Planner and simulator/controller joint-order verification.
- Conversion between planner qpos and execution qpos.
- Execution through a stable joint-position controller or equivalent low-level
  controller.
- Per-arm gripper command bookkeeping in the downstream environment.
- Optional target-grasp visualization markers.
- Synchronization of the internal MPlib robot state with the live robot or
  simulator state before planning.

This lets a planned path be executed directly as simulator or robot-control
actions rather than remaining an offline kinematic artifact.

## Example Tasks

The planner was used to script demonstrations for bimanual Colosseum V2 tasks.

| Task family | What it demonstrates | Video |
| --- | --- | --- |
| Cube handoff | Dual-arm target-pose planning, single-arm screw motion, and coordinated gripper timing. | <video src="https://github.com/user-attachments/assets/e0437622-49c0-42f8-b0a8-d7b60150be64" width="100%" controls></video>|
| Drawer interaction | Collision-aware approach planning and local Cartesian motion around articulated objects. | <video src="https://github.com/user-attachments/assets/e221d32f-ede8-448e-b390-bceaf76b9fa6" width="100%" controls></video>|
| Cube stacking | Alternating single-arm and dual-arm plans for long-horizon pick-and-place. | <video src="https://github.com/user-attachments/assets/be728c13-c610-4ba6-91b1-25bf98383194" width="100%" controls></video>|
| Pen-cap manipulation | Fine-grained bimanual pose sequencing and gripper coordination. | <video src="https://github.com/user-attachments/assets/ce0fa9c2-ba46-4939-9388-2331aa76e16f" width="100%" controls></video>|
| Threading-style manipulation | Local Cartesian motion with simultaneous end-effector goals. | <video src="https://github.com/user-attachments/assets/44ba014c-a1a1-4486-a95b-0f3a947def1b" width="100%" controls></video>|
| Box pushing | Synchronized dual-arm Cartesian motion in a shared workspace. | <video src="https://github.com/user-attachments/assets/991d4559-7763-4c28-9adc-a86c2d0f1273" width="100%" controls></video>|

These tasks require a mix of global planning, local Cartesian motion, gripper
timing, collision reasoning, and two-arm coordination.

## Design Decisions

### Treat the Two Arms as One Planning Problem

Instead of planning independently for each arm and trying to synchronize the
results afterward, the planner models both arms in one configuration vector.
This enables cross-arm collision checking and simultaneous end-effector goals.

### Use Pinocchio for SE(3)-Level Reasoning

Pinocchio provides a clean representation for task-space errors, Jacobians, and
rigid-body transforms. This makes the implementation suitable for simultaneous
dual-arm IK and local Cartesian motion.

### Combine Global and Local Planning

RRTConnect is used for larger collision-aware moves, while differential IK(screw motion) is
used for short Cartesian motions. This hybrid design is practical for
long-horizon manipulation, where no single planner is best for every segment.

### Preserve Execution Compatibility

The planner is written around the realities of downstream execution:
joint-order mismatch, gripper mimic joints, finite control timesteps, and
controller-specific action layouts.

## Current Limitations

- The implementation is specialized for a fixed-base dual Franka Panda setup.
- Some task-level behavior still requires hand-designed grasp poses and
  waypoint sequences.
- Two-arm constrained transport while both grippers rigidly hold the same object
  is experimental and is not presented here as a supported capability.
- The planner is primarily designed for demonstration generation rather than
  real-time replanning.

## Research and Engineering Relevance

This project demonstrates experience with:

- Sampling-based motion planning with OMPL/MPlib.
- Rigid-body kinematics and Jacobian-based control with Pinocchio.
- Collision-aware planning for multi-arm systems.
- Coordinated bimanual task-space goals in SE(3).
- Trajectory smoothing and time parameterization.
- Robotics simulator integration and dataset generation.
- Building reusable motion primitives for long-horizon manipulation.

The implementation sits at the intersection of motion planning, robot learning,
and simulation infrastructure: it turns low-level planning components into a
working system capable of producing bimanual manipulation demonstrations for
VLA benchmark tasks.

