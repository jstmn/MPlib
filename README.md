# MPlib: Bimanual Motion Planning Extension

![Bimanual Demo](https://raw.githubusercontent.com/jstmn/MPlib/Prajwal/demos/2026-01-22_11%3A05%3A34___n%3A8.mp4)
![Bimanual Demo 2](https://raw.githubusercontent.com/jstmn/MPlib/Prajwal/demos/2026-01-23_08%3A17%3A34___n%3A4.mp4)

**Fork extending MPlib for bimanual manipulation planning, integrated with ManiSkill**

## 🚀 Key Contributions
- **Bimanual manipulation support** using Pinocchio (dynamics), OMPL (planning), FCL (collision)
- **Full ManiSkill integration** for GPU-accelerated benchmarks
- **Pinocchio/OMPL/FCL integration** for dual-arm motion planning

## Installation
```bash
pip install mplib
pip install pinocchio ompl fcl  # Bimanual dependencies
