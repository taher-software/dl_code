from enum import Enum


class ChallengeType(Enum):
    Mfa_Otp = 'Mfa_Otp'
    Confirm_Email = 'Confirm_Email'
    Reset_Password = 'Reset_Password'
