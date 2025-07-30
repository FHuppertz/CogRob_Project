# CogRob_Project

## URDF Model Modifications

This project uses a modified version of the KUKA iiwa robot URDF model. Two key changes were made to the original model to make it suitable for mobile manipulation tasks:

1. **Base Link Mass Increase**: The mass of link_0 (base) was increased from 0.0 kg to 20.0 kg to provide a stable base for the robot arm when mounted on a mobile platform.

2. **First Joint Type Change**: Joint_1 was changed from a "revolute" joint with position limits to a "continuous" joint without position limits, allowing unlimited rotation which is essential for a mobile manipulator.

### Specific Changes

**File**: `model_2.urdf`

**Change 1 - Base Link Mass**:
- **Line 67**: `<mass value="0.0"/>` → `<mass value="20.0"/>`
- This increases the stability of the robot's base.

**Change 2 - First Joint Type**:
- **Line 85**: `<joint name="lbr_iiwa_joint_1" type="revolute">` → `<joint name="lbr_iiwa_joint_1" type="continuous">`
- **Lines 90**: Removed position limits:
  ```xml
  <limit effort="300" lower="-2.96705972839" upper="2.96705972839" velocity="10"/>
  ```
  Changed to:
  ```xml
  <limit effort="300" velocity="10"/>
  ```

These modifications are essential for the proper functioning of the mobile manipulator in simulation.

### File Path

The modified URDF file needs to be saved in your environment's site-packages directory:
`./env/lib64/python3.12/site-packages/pybullet_data/kuka_iiwa/model_2.urdf`
