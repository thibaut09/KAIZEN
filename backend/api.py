import os
import sys
import json
import uuid
import shutil
import asyncio
import hashlib
from typing import Optional, Any
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from eth_account import Account
from eth_account.messages import encode_defunct
from eth_abi.packed import encode_packed
from web3 import Web3

project_root = Path(__file__).resolve().parent.parent
mint_dir = project_root / "frontend/public/minted_nfts"
temp_dir = project_root / "temp_uploads"
dataset_dir = project_root / "dataset/full_dataset"

load_dotenv(dotenv_path=project_root / ".env")
oracle_private_key = os.getenv("ORACLE_PRIVATE_KEY")
oracle_address = os.getenv("ORACLE_ADDRESS")

sys.path.insert(0, str(project_root / "ml" / "models"))
from embedding import kaizen

allowed_exts = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.jfif'}
allowed_mimes = {'image/png', 'image/jpeg', 'image/webp', 'image/gif'}

app = FastAPI(title="KAIZEN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/training_data", StaticFiles(directory=str(dataset_dir)), name="training_data")

print("[kaizen] starting ...")
oracle = kaizen(mode="inference")
print(f"[kaizen] is active. signer: {oracle_address}")


class AuditResponse(BaseModel):
    status: str
    score: float
    match: Optional[str] = None
    match_name: Optional[str] = None
    entropy: Optional[float] = None
    s_sig: Optional[float] = None
    s_dino: Optional[float] = None
    error: Optional[str] = None
    image_hash: Optional[str] = None
    signature: Optional[str] = None
    tokenURI: Optional[str] = None
    similarity_score_int: Optional[int] = None
    oracle_address: Optional[str] = oracle_address

class ConfirmRequest(BaseModel):
    image_hash: str
    tx_hash: str
    chain_id: int


@app.post("/audit", response_model=AuditResponse)
async def audit_image(file: UploadFile = File(...), user_address: Optional[str] = Form(None)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    
    if (ext not in allowed_exts) and (file.content_type not in allowed_mimes):
        raise HTTPException(status_code=400, detail="invalid image format")

    safe_name = f"{uuid.uuid4().hex}{ext or '.png'}"
    temp_path = temp_dir / safe_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        img_bytes = temp_path.read_bytes()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        
        if (mint_dir / f"{img_hash}.json").exists():
            return AuditResponse(status="REJECTED", score=1.0, error="artwork already minted")
            
        report = await asyncio.to_thread(oracle.analyze_image, str(temp_path))
        if not report:
            return AuditResponse(status="ERROR", score=0, error="processing failed")
            
        out = {
            "status": report.get("status", "ERROR"),
            "score": report.get("score", 0.0),
            "match": report.get("match"),
            "match_name": report.get("match_name"),
            "entropy": report.get("entropy"),
            "s_sig": report.get("s_sig"),
            "s_dino": report.get("s_dino")
        }
        
        if out["status"] == "APPROVED" and user_address and oracle_private_key:
            out["image_hash"] = img_hash
            out["similarity_score_int"] = int(out["score"] * 10000)
            out["tokenURI"] = f"/minted_nfts/{img_hash}.json"
            
            mint_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_path, mint_dir / f"{img_hash}.png")
            
            metadata = {
                "name": f"Original Artwork #{img_hash[:8]}",
                "description": "An original digital artwork verified  by the KAIZEN",
                "image": f"/minted_nfts/{img_hash}.png",
                "attributes": [
                    {"trait_type": "KAIZEN Originality Verdict", "value": "APPROVED"},
                    {"trait_type": "Shannon Entropy", "value": out["entropy"]},
                    {"trait_type": "Siglip 2 Score", "value": out["s_sig"]},
                    {"trait_type": "Dinov2 Score", "value": out["s_dino"]}
                ]
            }
            (mint_dir / f"{img_hash}.json").write_text(json.dumps(metadata, indent=2))
            
            digest = Web3.keccak(encode_packed(
                ['address', 'string', 'string', 'uint256'],
                [Web3.to_checksum_address(user_address), out["tokenURI"], img_hash, out["similarity_score_int"]]
            ))
            out["signature"] = Account.sign_message(encode_defunct(digest), private_key=oracle_private_key).signature.hex()

        return AuditResponse(**out)
        
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/minted")
def get_minted_nfts():
    if not mint_dir.exists():
        return []
    
    return [
        {
            "tokenId": p.stem[:8].upper(),
            "imageHash": p.stem,
            "score": "0.00%",
            "txHash": "Local Development Node"
        } 
        for p in sorted(mint_dir.glob("*.json"), reverse=True)
    ]


@app.post("/confirm")
def confirm_mint(req: ConfirmRequest):
    filepath = mint_dir / f"{req.image_hash}.json"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="metadata file not found")
        
    try:
        data = json.loads(filepath.read_text())
        data["txHash"] = req.tx_hash
        data["chainId"] = req.chain_id
        
        network_name = "Alastria Red B" if req.chain_id == 2020 else "Hardhat Localhost" if req.chain_id == 31337 else f"Chain {req.chain_id}"
        
        attributes = data.get("attributes", [])
        attributes = [attr for attr in attributes if attr["trait_type"] not in ["Network", "Blockchain Transaction"]]
        
        attributes.extend([
            {"trait_type": "Network", "value": network_name},
            {"trait_type": "Blockchain Transaction", "value": req.tx_hash}
        ])
        data["attributes"] = attributes
        
        filepath.write_text(json.dumps(data, indent=2))
        return {"status": "SUCCESS"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to update metadata: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
