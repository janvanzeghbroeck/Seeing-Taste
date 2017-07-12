
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders
import os



def send_email(fromaddr,psw,toaddr,subject,body,filename_lst):

    msg = MIMEMultipart()

    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # attaching files
    for filename in filename_lst:

        attachment = open(filename, "rb")

        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)

        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, psw)
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
    print 'send email from {} to {} attached with {}'.format(fromaddr,toaddr,filename)

if __name__ == '__main__':
    fromaddr = "vanzeghbroeck.jan@gmail.com" #"YOUR EMAIL"
    # toaddr = 'vanzeghb@gmail.com' #"EMAIL ADDRESS YOU SEND TO"
    toaddr = 'steve.v.iannaccone@gmail.com'
    psw = os.environ['EMAIL_PSW_DOTJAN'] #get password for email
    body = "ITS LOOKING AT YOU!!"
    subject = "SUBJECT OF THE EMAIL"
    filename = "pint.png"

    send_email(fromaddr,psw,toaddr,subject,body,filename)
