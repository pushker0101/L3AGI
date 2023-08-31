from pydantic import BaseModel, UUID4
from typing import List, Optional
from api.user import User
from api.account import Account
from typing import List, Optional

class LoginInput(BaseModel):
    email: str
    password: str
    
class GoogleInput(BaseModel):
    email: str
    password: str
    
class GithubInput(BaseModel):
    email: str
    password: str
    
class RegisterInput(BaseModel):
    name: str
    email: str
    account_name: str

class UserAccount(BaseModel):
    user: User
    account: Account
    