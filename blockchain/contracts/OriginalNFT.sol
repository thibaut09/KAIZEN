pragma solidity ^0.8.24;

import { ERC721 } from "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import { ERC721URIStorage } from "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import { Ownable } from "@openzeppelin/contracts/access/Ownable.sol";
import { ECDSA } from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import { MessageHashUtils } from "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";

contract original_nft is ERC721URIStorage, Ownable {
    using ECDSA for bytes32;

    uint256 private next_token_id;
    address public ai_oracle;
    
    mapping(string => bool) public is_minted;

    event oracle_updated(address indexed old_oracle, address indexed new_oracle);
    event artwork_minted(address indexed recipient, uint256 indexed token_id, string image_hash, uint256 score);

    error invalid_oracle();
    error already_minted();
    error unauthorized_signature();

    constructor(address initial_oracle) ERC721("Original Artwork", "ORIG") Ownable(msg.sender) {
        if (initial_oracle == address(0)) revert invalid_oracle();
        ai_oracle = initial_oracle;
    }

    function set_oracle(address new_oracle) external onlyOwner {
        if (new_oracle == address(0)) revert invalid_oracle();
        
        address old_oracle = ai_oracle;
        ai_oracle = new_oracle;
        
        emit oracle_updated(old_oracle, new_oracle);
    }

    function mint_artwork(
        address recipient,
        string calldata token_uri,
        string calldata image_hash,
        uint256 score,
        bytes calldata signature
    ) external {
        if (is_minted[image_hash]) revert already_minted();

        bytes32 digest = MessageHashUtils.toEthSignedMessageHash(
            keccak256(abi.encodePacked(recipient, token_uri, image_hash, score))
        );
        
        if (digest.recover(signature) != ai_oracle) revert unauthorized_signature();

        is_minted[image_hash] = true;
        
        uint256 token_id = next_token_id;
        next_token_id++;

        _safeMint(recipient, token_id);
        _setTokenURI(token_id, token_uri);
        
        emit artwork_minted(recipient, token_id, image_hash, score);
    }
}
