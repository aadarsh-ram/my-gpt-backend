import pika
import json

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', heartbeat=600))
channel = connection.channel()

channel.queue_declare(queue='gpt-response', durable=True)
channel.queue_declare(queue='gpt-send', durable=True)

# Summary test
def summary_test():
    send_summary_json = {
        'type' : 'summary',
        'content' : '/home/aadarsh/src/llm-sih-test/my-gpt-backend/samples/trump-article.pdf'
    }

    channel.basic_publish(exchange='', routing_key='gpt-send', body=json.dumps(send_summary_json))
    print("[x] Sent PDF")

# Grammar test
def grammar_test():    
    send_grammar_json = {
        'type' : 'grammar',
        'content' : "did you no that bats are mammals. we no they are mammals just lik us becaus they are warm blooded they are the only mammals that no how to fly bats are Nocturnal which means thay sleep during the day and are awak at nite?."
    }
    channel.basic_publish(exchange='', routing_key='gpt-send', body=json.dumps(send_grammar_json))
    print("[x] Sent wrong text")

# Upload test
def upload_test(path):
    send_upload_json = {
        'type' : 'upload',
        'content' : path
    }
    channel.basic_publish(exchange='', routing_key='gpt-send', body=json.dumps(send_upload_json))
    print ("[x] Sent upload file")

# Chat test
def chat_test():
    send_chat_json1 = {
        'type' : 'chat',
        'query' : 'What is the date of birth of Sachin Tendulkar?',
        'chat_history' : []
    }
    channel.basic_publish(exchange='', routing_key='gpt-send', body=json.dumps(send_chat_json1))

    send_chat_json2 = {
        'type' : 'chat',
        'query' : 'Among those, how many were US citizens?',
        'chat_history' : [
            'How many captives had Hamas freed?', " Hamas had freed four captives, including two US citizens."
        ]
    }
    channel.basic_publish(exchange='', routing_key='gpt-send', body=json.dumps(send_chat_json2))

# Sample invocation
grammar_test()

def callback(ch, method, properties, body):
    print (f"Received {body}")

channel.basic_consume(queue='gpt-response', on_message_callback=callback, auto_ack=True)
channel.start_consuming()