import pika
import json

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', heartbeat=600))
channel = connection.channel()

channel.queue_declare(queue='gpt-response', durable=True)
channel.queue_declare(queue='gpt-send', durable=True)

# Summary test
# send_summary_json = {
#     'type' : 'summary',
#     'content' : '/home/aadarsh/src/llm-sih-test/my-gpt-backend/samples/trump-article.pdf'
# }

# channel.basic_publish(exchange='', routing_key='gpt-send', body=json.dumps(send_summary_json))
# print("[x] Sent PDF")

# Grammar test
send_grammar_json = {
    'type' : 'grammar',
    'content' : "did you no that bats are mammals. we no they are mammals just lik us becaus they are warm blooded they are the only mammals that no how to fly bats are Nocturnal which means thay sleep during the day and are awak at nite?."
}
channel.basic_publish(exchange='', routing_key='gpt-send', body=json.dumps(send_grammar_json))
print("[x] Sent wrong text")

def callback(ch, method, properties, body):
    print (f"Received {body}")

channel.basic_consume(queue='gpt-response', on_message_callback=callback, auto_ack=True)
channel.start_consuming()