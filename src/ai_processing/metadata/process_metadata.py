import json
from app.beta.models.video import Video
import datetime
import numpy as np
import os
import random
import re
from app import app
from config import Config

# please do not remove this line
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

       
       
