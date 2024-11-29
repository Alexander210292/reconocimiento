STATUS = "LOCAL2"

if STATUS == "LOCAL":
    SERVER = 'http://127.0.0.1:8000/'
else:
    SERVER = 'http://177.222.109.127/'
    
LOGIN_ENDPOINT = SERVER + "api/login/"
TRANSACTION_ENDPOINT = SERVER + "api/transactions/"

    