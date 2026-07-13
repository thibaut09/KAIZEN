import os
import sys
import json
from pathlib import Path

import solcx
from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

network = "alastria" if len(sys.argv) > 1 and sys.argv[1] == "--alastria" else "hardhat"
rpc_url = "http://sinbad2.ujaen.es:8012" if network == "alastria" else "http://127.0.0.1:8545"

print(f"[KAIZEN] connecting to {network} at {rpc_url}...")
w3 = Web3(Web3.HTTPProvider(rpc_url))

if not w3.is_connected():
    print(f"error: failed to connect to provider at {rpc_url}")
    sys.exit(1)

oracle_pk = os.getenv("ORACLE_PRIVATE_KEY")
if not oracle_pk:
    oracle = Account.create()
    with open(project_root / ".env", "a") as env_file:
        env_file.write(f"\nORACLE_PRIVATE_KEY={oracle.key.hex()}\nORACLE_ADDRESS={oracle.address}\n")
    oracle_pk = oracle.key.hex()
    print("[KAIZEN] generated new oracle keypair")

oracle = Account.from_key(oracle_pk)
deployer_pk = os.getenv("DEPLOYER_PRIVATE_KEY") or oracle_pk
deployer = Account.from_key(deployer_pk)

print(f"[KAIZEN] deployer address: {deployer.address}")
print("[KAIZEN] compiling smart contract...")

solcx.install_solc("0.8.24")
solcx.set_solc_version("0.8.24")

compiled = solcx.compile_files(
    ["contracts/OriginalNFT.sol"],
    output_values=["abi", "bin"],
    import_remappings=["@openzeppelin/=node_modules/@openzeppelin/"],
    evm_version="cancun"
)

target = next((k for k in compiled.keys() if k.endswith(":original_nft")), None)
if not target:
    raise ValueError("contract compilation failed")

abi = compiled[target]["abi"]
bytecode = compiled[target]["bin"]
contract = w3.eth.contract(abi=abi, bytecode=bytecode)

print(f"[KAIZEN] deploying to {network}...")

if network == "alastria":
    construct_txn = contract.constructor(oracle.address).build_transaction({
        "from": deployer.address,
        "nonce": w3.eth.get_transaction_count(deployer.address),
        "gas": 3000000,
        "gasPrice": w3.eth.gas_price or 0,
        "chainId": 2020
    })
    signed_txn = w3.eth.account.sign_transaction(construct_txn, private_key=deployer_pk)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
else:
    deployer_address = w3.eth.accounts[0]
    tx_hash = contract.constructor(oracle.address).transact({"from": deployer_address})

print(f"[KAIZEN] tx sent. hash: {tx_hash.hex()}. waiting for confirmation...")
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print(f"[KAIZEN] deployed at {receipt.contractAddress}")

config_path = project_root / "frontend/src/config/contract.json"
config_path.parent.mkdir(parents=True, exist_ok=True)

existing_config = {}
if config_path.exists():
    try:
        existing_config = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        pass

addresses = existing_config.get("addresses", {})
addresses[str(w3.eth.chain_id)] = receipt.contractAddress

updated_config = {
    "address": receipt.contractAddress,
    "addresses": addresses,
    "abi": abi
}

config_path.write_text(json.dumps(updated_config, indent=2))

