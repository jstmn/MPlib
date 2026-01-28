# MPlib: Bimanual Motion Planning Extension

[![Bimanual Demo](https://github.com/user-attachments/assets/18f77352-2b13-4f1b-a25b-4b8c01773f53)](https://github.com/user-attachments/assets/18f77352-2b13-4f1b-a25b-4b8c01773f53)

**Fork extending MPlib for bimanual manipulation planning, integrated with ManiSkill for Colosseum V2 benchmarks (RSS 2026 submission).**

## 🚀 Key Contributions
- **Bimanual manipulation support** using Pinocchio (dynamics), OMPL (planning), FCL (collision)
- **Full ManiSkill integration** for GPU-accelerated benchmarks
- **Pinocchio/OMPL/FCL integration** for dual-arm motion planning

## Installation
```bash
pip install mplib
pip install pinocchio ompl fcl  # Bimanual dependencies
