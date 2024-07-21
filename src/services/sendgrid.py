import main as app
from src.models.user import User
import datetime
import logging
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import urllib

class SendGridService:

  @staticmethod
  def send_confirm_email(user):
    template_id = 'd-671dc725c686461399f4b443e9cdc9af'
    dynamic_template_data = {
      'email_confirmation': f"{app.Config.WEBSITE_URL}confirm-email/{user.get_confirm_email_token()}",
      'first_name': user.first_name
    }
    SendGridService.send_mail(user.email, template_id, dynamic_template_data)

  @staticmethod
  def send_download_complete(user, downloads):
    template_id = 'd-fa8bbcb6850b445e8cc31e9ac5d3ec28'
    dynamic_template_data = {
      'first_name': user.first_name,
      'website_link': app.Config.WEBSITE_URL,
      'downloads': downloads
    }
    SendGridService.send_mail(user.email, template_id, dynamic_template_data)

  @staticmethod
  def welcome_aboard(user):
    template_id = 'd-527922e6ce7f4631b180952ea3b3d002'
    dynamic_template_data = {
      'first_name': user.first_name
    }
    SendGridService.send_mail(user.email, template_id, dynamic_template_data)

  @staticmethod
  def send_otp_email(user, otp):
    template_id = 'd-692c35ee84db489da9dec054d4bdf6ab'
    dynamic_template_data = {
      'first_name': user.first_name,
      'otp_code': otp
    }
    SendGridService.send_mail(user.email, template_id, dynamic_template_data)

  @staticmethod
  def send_video_processed_confirmation(user, videos):
    template_id = 'd-43ac849374d146079168e5a0dda6b04e'
    dynamic_template_data = {
      'first_name': user.first_name,
      'videos': videos
    }
    SendGridService.send_mail(user.email, template_id, dynamic_template_data)

  @staticmethod
  def send_password_reset_email(user):
    template_id = 'd-7fe34556832e49029c6ff527bd9cc9c8'
    dynamic_template_data = {
      'password_reset_link': f"{app.Config.WEBSITE_URL}reset-password/{user.get_reset_password_token()}",
      'first_name': user.first_name
    }
    SendGridService.send_mail(user.email, template_id, dynamic_template_data)

  @staticmethod
  def send_password_reset_complete_email(user):
    template_id = 'd-7b7391c3a9364e929bddb2a257f4b0aa'
    dynamic_template_data = {
      'website_url': app.Config.WEBSITE_URL
    }
    SendGridService.send_mail(user.email, template_id, dynamic_template_data)

  @staticmethod
  def send_invite(user, email, msg):
    template_id = 'd-87bf1ca6ed1d4a828ee659c2d38930f8'
    dynamic_template_data = {
      'signup_url': f'{app.Config.WEBSITE_URL}auth/register',
      'first_name': user.first_name,
      'last_name': user.last_name,
      'invite_message': msg
    }
    SendGridService.send_mail(email, template_id, dynamic_template_data)

  @staticmethod
  def send_collaborator_invite(user, email, msg):
    existing_user = User.query.filter_by(email=email).first()
    signup_url = f'{app.Config.WEBSITE_URL}auth/register?em={urllib.parse.quote(email)}' if existing_user is None else f'{app.Config.WEBSITE_URL}auth/login'

    template_id = 'd-f4e4ee9e0b034bebb396840b479af912'
    dynamic_template_data = {
      'signup_url': signup_url,
      'first_name': user.first_name,
      'last_name': user.last_name,
      'invite_message': msg
    }
    SendGridService.send_mail(email, template_id, dynamic_template_data)

  """Send Payment Success"""
  @staticmethod
  def send_payment_success(email, receipt_url):
    user = User.query.filter_by(email=email).first()

    template_id = 'd-abdee67648a9472697936170df94aa56'
    dynamic_template_data = {
      'first_name': user.first_name,
      'receipt_url': receipt_url
    }
    SendGridService.send_mail(user.email, template_id, dynamic_template_data)

  @staticmethod
  def send_downgrade_confirmation(email, final_billing_date):
    date = datetime.datetime.fromtimestamp(final_billing_date)
    user = User.query.filter_by(email=email).first()
    template_id = 'd-dd50d8ecb3404dc4b9b23d25e6a9c0ce'
    dynamic_template_data = {
      'first_name': user.first_name,
      'final_billing_date': date.strftime("%c")
    }
    SendGridService.send_mail(user.email, template_id, dynamic_template_data)

  @staticmethod
  def send_mail(to_emails, template_id, dynamic_template_data):
    dynamic_template_data['Sender_Name'] = app.Config.SENDGRID_SENDER_NAME
    dynamic_template_data['Sender_Address'] = app.Config.SENDGRID_SENDER_ADDRESS
    dynamic_template_data['Sender_City'] = app.Config.SENDGRID_SENDER_CITY
    dynamic_template_data['Sender_State'] = app.Config.SENDGRID_SENDER_STATE
    dynamic_template_data['Sender_Zip'] = app.Config.SENDGRID_SENDER_ZIP
    message = Mail(
        from_email=app.Config.ADMINS[0],
        to_emails=to_emails)
    message.dynamic_template_data = dynamic_template_data
    message.template_id = template_id
    try:
        sg = SendGridAPIClient(app.Config.SENDGRID_API_KEY)
        response = sg.send(message)
    except Exception as e:
        logging.error('Send failed', e.message)