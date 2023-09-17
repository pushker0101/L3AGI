from typing import List
import json
from fastapi import APIRouter, HTTPException, Depends
from fastapi_sqlalchemy import db
from uuid import uuid4
import os
from pathlib import Path

from models.config import ConfigModel
from typings.config import ConfigOutput, ConfigInput, ConfigQueryParams
from utils.auth import authenticate
from typings.auth import UserAccount
from utils.configuration import convert_configs_to_config_list, convert_model_to_response
from exceptions import ConfigNotFoundException
from datasources.base import DatasourceEnvKeyType
from services.aws_s3 import AWSS3Service
from pdf.pdf_async import L3FileRetriever
router = APIRouter()

@router.post("", status_code=201, response_model=ConfigOutput)
def create_config(config: ConfigInput, auth: UserAccount = Depends(authenticate)) -> ConfigOutput:
    """
    Create a new config with configurations.

    Args:
        config (ConfigInput): Data for creating a new config with configurations.
        auth (UserAccount): Authenticated user account.

    Returns:
        ConfigOutput: Created config object.
    """
    # Consider adding try-except for error handling during creation if needed
    db_config = ConfigModel.create_config(db, config=config, user=auth.user, account=auth.account)

    # Index the file documents
    if config.datasource_id and config.key_type == DatasourceEnvKeyType.FILES.value:
        file_urls = json.loads(config.value)

        tmp_dir = f"pdf/tmp/{config.datasource_id}"
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

        for file_url in file_urls:
            key = AWSS3Service.get_key_from_public_url(file_url)
            file_path = f"{tmp_dir}/{uuid4()}.pdf"
            absolute_path = Path(file_path).resolve()
            AWSS3Service.download_file(key, absolute_path)

        retriever = L3FileRetriever(Path(tmp_dir).resolve())
        retriever.query("summarize the documents")


    return convert_model_to_response(ConfigModel.get_config_by_id(db, db_config.id, auth.account))

@router.put("/{id}", status_code=200, response_model=ConfigOutput)  # Changed status code to 200
def update_config(id: str, config: ConfigInput, auth: UserAccount = Depends(authenticate)) -> ConfigOutput:
    """
    Update an existing config with configurations.

    Args:
        id (str): ID of the config to update.
        config (ConfigInput): Data for updating the config with configurations.
        auth (UserAccount): Authenticated user account.

    Returns:
        ConfigOutput: Updated config object.
    """
    try:
        db_config = ConfigModel.update_config(db, 
                                           id=id, 
                                           config=config, 
                                           user=auth.user, 
                                           account=auth.account)
        return convert_model_to_response(ConfigModel.get_config_by_id(db, db_config.id, auth.account))
    
    except ConfigNotFoundException:
        raise HTTPException(status_code=404, detail="Config not found")

@router.get("", response_model=List[ConfigOutput])
def get_configs(auth: UserAccount = Depends(authenticate),
                params: ConfigQueryParams = Depends()
                ) -> List[ConfigOutput]:
    """
    Get all configs by account ID.

    Args:
        auth (UserAccount): Authenticated user account.

    Returns:
        List[ConfigOutput]: List of configs associated with the account.
    """
    # print(params)
    db_configs = ConfigModel.get_configs(db=db, query=params, account=auth.account)
    return convert_configs_to_config_list(db_configs)

@router.get("/{id}", response_model=ConfigOutput)
def get_config_by_id(id: str, 
                     auth: UserAccount = Depends(authenticate)
                     ) -> ConfigOutput:
    """
    Get a config by its ID.

    Args:
        id (str): ID of the config.
        auth (UserAccount): Authenticated user account.

    Returns:
        ConfigOutput: Config associated with the given ID.
    """
    db_config = ConfigModel.get_config_by_id(db, config_id=id, account=auth.account)
    
    if not db_config or db_config.is_deleted:
        raise HTTPException(status_code=404, detail="Config not found")  # Ensure consistent case in error messages

    return convert_model_to_response(db_config)

@router.delete("/{config_id}", status_code=200)  # Changed status code to 204
def delete_config(config_id: str, auth: UserAccount = Depends(authenticate)):
    """
    Delete a config by its ID. Performs a soft delete by updating the is_deleted flag.

    Args:
        config_id (str): ID of the config to delete.
        auth (UserAccount): Authenticated user account.

    Returns:
        dict: A dictionary indicating the success or failure of the deletion.
    """
    try:
        ConfigModel.delete_by_id(db, config_id=config_id, account=auth.account)
        return { "message": "Config deleted successfully" }

    except ConfigNotFoundException:
        raise HTTPException(status_code=404, detail="Config not found")

