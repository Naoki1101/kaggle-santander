import requests

def send_message(message):
    line_token = 'QpiJEDvadb4HAEhm0xiRkij6VVFcFiXKFxoExA1OlJw'
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)
