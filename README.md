KAIZEN

machine learning pipeline designed to detect asset plagiarism in nft collections. it breaks the sibling paradox by accurately identifying manipulated artwork without falsely flagging legitimate assets from the same collection. uses siglip 2 and dinov2 for semantic and structural feature extraction.

setup
- python 3.9+
- node v18+

install
```bash
git clone https://github.com/thibaut09/KAIZEN.git
cd KAIZEN

pip install -r requirements.txt

cd frontend && npm install
cd ../blockchain && npm install
```

run
```bash
start_nft_system.bat
```

manual boot
1. blockchain: `npx hardhat node`
2. oracle api: `uvicorn backend.api:app --reload --port 8000`
3. interface: `cd frontend && npm run dev`


