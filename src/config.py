import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):

    #ADMIN    
    API_URL=os.environ.get('API_URL') or 'http://127.0.0.1:5000/api/v1/'
    SESSION_COOKIE_HTTPONLY = False
    SESSION_COOKIE_SAMESITE = 'none'
    SESSION_COOKIE_SECURE = True
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    # SQLALCHEMY_ECHO = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MY_SQL_POOL_SIZE = os.environ.get('MY_SQL_POOL_SIZE', None)
    MY_SQL_MAX_OVERFLOW = os.environ.get('MY_SQL_MAX_OVERFLOW', None)
    MAIL_SERVER = 'smtp.sendgrid.net'
    MAIL_PORT = 465
    MAIL_USE_TLS = False
    MAIL_USE_SSL = True
    MAIL_USERNAME = 'apikey'
    MAIL_PASSWORD = ''
    ADMINS = ['support@imaginario.ai']
    #for Test
    SEGMENT_API_KEY = os.environ.get('SEGMENT_API_KEY') or 'mZUmSpkmZKIBZW1uiHMGlh06fDKYLrCG'
    #for prod
    #SEGMENT_API_KEY = '2yoWxtHpU2FwenwoJ8njKoc8atOUKKK9'
    ROOT_FOLDER = os.environ.get('ROOT_FOLDER') or '/home/loukkal/Imaginario-app'
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://'
    RQ_REDIS_URL = os.environ.get('REDIS_URL') or 'redis://'
    ROOT_APP_FOLDER = os.environ.get('ROOT_APP_FOLDER') or '/home/loukkal/Imaginario-app'
    WEBSITE_URL = os.environ.get('WEBSITE_URL') or os.environ.get('IMGN_WEBSITE_ADDRESS') or 'https://app.imaginario.ai/'
    SOCKET_URI = os.environ.get('SOCKET_URI') or "ws://localhost:5001"
    PAGINATION_ITEMS = 8

    IMAGINARIO_WEB_URL= os.environ.get('IMGN_WEBSITE_ADDRESS')
    IMAGINARIO_AUTH_JWT_SIGNING_SECRETE=os.environ.get('IMAGINARIO_AUTH_JWT_SIGNING_SECRETE') or 'ABDEL'
    IMAGINARIO_AUTH_GOD_MODE_API_KEY=os.environ.get('IMAGINARIO_AUTH_GOD_MODE_API_KEY') or 'ABDEL'
    IMAGINARIO_FILESYSTEM_ROOT_DIR=os.environ.get('ROOT_APP_FOLDER') or '/home/loukkal/Imaginario-app'
    
    #Stripe
    STRIPE_API_KEY =  os.environ.get('STRIPE_API_KEY') or 'sk_test_51LrioSJ19vUdJG79fTCm7B1DyKqWok8iyQNhz7iILYQn3yLmJh9vNpNKvViBa8svp3mX16KIhOJDWXgrdD7DVi8000Gpe5SnBQ'
    # If you are testing with the CLI, find the secret by running 'stripe listen'
    STRIPE_WEBHOOK = os.environ.get('STRIPE_WEBHOOK') or 'whsec_00a4769d8de68e644700d1280d58f2c303e2b27406fed6a6f484584982071e5a'
    
    #Google
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID') or '779544015607-ceksdts3cnn1slrm0c8fs7nskem48chv.apps.googleusercontent.com'
    
    #Sendgrid
    SENDGRID_API_KEY=os.environ.get('SENDGRID_API_KEY') 
    SENDGRID_SENDER_NAME=os.environ.get('SENDGRID_SENDER_NAME') or "Imaginario AI"
    SENDGRID_SENDER_ADDRESS=os.environ.get('SENDGRID_SENDER_ADDRESS') or ""
    SENDGRID_SENDER_CITY=os.environ.get('SENDGRID_SENDER_CITY') or "Philadelphia"
    SENDGRID_SENDER_STATE=os.environ.get('SENDGRID_SENDER_STATE') or "Pennsylvania, United States"
    SENDGRID_SENDER_ZIP=os.environ.get('SENDGRID_SENDER_ZIP') or "19120"
    PORT=os.environ.get('IM_PORT') or os.environ.get('PORT') or 5000
    TEST_WORKERS=os.environ.get("TEST_WORKERS") or '0'
    ENTERPRISE_CLIENT=os.environ.get('ENTERPRISE_CLIENT') or '0'
    
     #day * hours * minutes * seconds
    MFA_AUTH_FREQUENCY = 7 * 24 * 60 * 60
    MFA_ENABLED = os.environ.get('MFA_ENABLED') or '0'
    MFA_AUTHENTICATOR_ENABLED = os.environ.get('MFA_AUTHENTICATOR_ENABLED') or '0'
    DOWNLOAD_TRAINING_ENABLED = os.environ.get('DOWNLOAD_TRAINING_ENABLED') or '0'
    TRAINING_BUCKET_NAME = os.environ.get('TRAINING_BUCKET_NAME') or '0'

    PROCESS_HLS = os.environ.get('PROCESS_HLS') or False
    
    #Pipedrive
    pipedrive_api_key = os.environ.get('pipedrive_api_key') or "ea031255503a59b0ceb1d8bf77f96aa904ae5148"
    pipedrive_last_seen_id = "da7960d08a809bb81278e825a5a86fc202d6b825"
    pipedrive_package_id = "b4ae31239dd0ac2a9a5510d016cf65373b20d78b"
    pipedrive_package_date_id = "fc11c3269b7cd09ce3e278397bd601e23032bdda"
    pipedrive_stage_id = "a03c331aba128b0fdea7f978ae3d5c7d57c5cc20"
    pipedrive_subscription_type_id = "278650ef3389b6f95c26fd0a6fd42606d46db257"

    #Transcoding
    mediaconvert_access_key = os.environ.get('mediaconvert_access_key') 
    mediaconvert_secret_key =  os.environ.get('mediaconvert_secret_key') 
    mediaconvert_region_name = os.environ.get('mediaconvert_region_name') or 'us-east-1'
    mediaconvert_arn_role =  os.environ.get('mediaconvert_arn_role') or 'arn:aws:iam::680512422523:role/ElementalMediaConvert'        
    mediaconvert_bucket_name = os.environ.get('mediaconvert_bucket_name') or 'mediaconvert-transcoding'

    #AI -dialogue processing
    stt_provider = os.environ.get('stt_provider') or "assembly"
    rev_api_key = os.environ.get('rev_api_key') or "02q8gPYRenqOdEyuuBlxFVqdJrdWUjX3PApwkcv8nG_oOVFcXVCfooJTSgqemf6olCtvSmwXgOcfJj7iT1LkD9bxXNNy4"
    assembly_api_key = os.environ.get('assembly_api_key') or "187dd5e3cbf5470989764c09417a306a"
    deepgram_api_key = os.environ.get('deepgram_api_key') or "9343538fb4a66bff5a7a0d691230a5ec346ecfe6"
    openai_api_key = os.environ.get('openai_api_key') 
    DIALOGUE_EMBEDDING_MODEL = os.environ.get('DIALOGUE_EMBEDDING_MODEL') or 'ALL-MINILM-L6-V2'    
        
    #AI -visual processing
    ai_models_weights_s3 = "ai-models-weights"
    PROCESS_CAST = os.environ.get('PROCESS_FACES') or False
    PROCESS_FACES = os.environ.get('PROCESS_FACES') or False
    VISUAL_EMBEDDING_MODEL = os.environ.get('VISUAL_EMBEDDING_MODEL') or 'CLIP_VIT_B_32'
    FACE_RECOGNITION_EMBEDDING_MODEL = os.environ.get('FACE_RECOGNITION_EMBEDDING_MODEL') or 'FACENET_VGG2'
    
    #AI -audio processing
    AUDIO_EMBEDDING_MODEL = os.environ.get('AUDIO_EMBEDDING_MODEL') or 'LARGER_CLAP_GENERAL'

    app_env = os.environ.get('app_env') or "prod"
    email_blacklist = os.environ.get('email_blacklist') or ['imaginario', 'appstrax']
    rekognition_access_key = os.environ.get('rekognition_access_key') 
    rekognition_secret_key =  os.environ.get('rekognition_secret_key') 
    rekognition_region_name = os.environ.get('rekognition_region_name') or 'us-east-1' 

    #Workers
    RQ_WORKER_CLASS='rq.worker.SimpleWorker'
    AUDIO_WORKER_NAME = os.environ.get('AUDIO_WORKER_NAME') or 'au'
    DOWNLOAD_WORKER_NAME = os.environ.get('DOWNLOAD_WORKER_NAME') or 'dl'
    INTEGRATION_WORKER_NAME = os.environ.get('INTEGRATION_WORKER_NAME') or 'in'
    TRANSCRIPT_WORKER_NAME = os.environ.get('TRANSCRIPT_WORKER_NAME') or 'tr'
    TRANSCODER_WORKER_NAME = os.environ.get('TRANSCODER_WORKER_NAME') or 'tc'
    PC_WORKER_NAME = os.environ.get('PC_WORKER_NAME') or 'pc'
    COMPUTING_DEVICE = os.environ.get('COMPUTING_DEVICE') or 'cuda'
    BURST_MODE = os.environ.get('BURST_MODE') or False

    VERSION = '2.1.677'

    ENTERPRISE_CLIENT=os.environ.get('ENTEPRRISE_CLIENT', '0')
    ENTEPRRISE_CLIENT=os.environ.get('ENTEPRRISE_CLIENT', '0')
    DEBUG=os.environ.get('DEBUG', '0')
    
    #GCP workers config
    TR_REDIS_URL = os.environ.get('TR_REDIS_URL') or 'redis://'
    PC_REDIS_URL = os.environ.get('PC_REDIS_URL') or 'redis://'
    AU_REDIS_URL = os.environ.get('AU_REDIS_URL') or 'redis://'
    DL_REDIS_URL = os.environ.get('DL_REDIS_URL') or 'redis://'
    IN_REDIS_URL = os.environ.get('IN_REDIS_URL') or 'redis://'
    GCP_ENV = os.getenv('GCP_ENV',False)
