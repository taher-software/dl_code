from enum import Enum


class AuthenticationType(Enum):
    Basic = 'Basic'
    EmailOtp = 'EmailOtp'
    AuthenticatorAppOtp = 'AuthenticatorAppOtp'
    Gmail = 'Gmail'
