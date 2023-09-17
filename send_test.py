import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', heartbeat=600))
channel = connection.channel()

channel.queue_declare(queue='gpt-response', durable=True)
channel.queue_declare(queue='gpt-send', durable=True)

channel.basic_publish(exchange='', routing_key='gpt-send', body='/home/aadarsh/src/llm-sih-test/my-gpt-backend/samples/pallative-care.pdf')
print("[x] Sent PDF")

def callback(ch, method, properties, body):
    print (f"Received summary {body}")

channel.basic_consume(queue='gpt-response', on_message_callback=callback, auto_ack=True)
channel.start_consuming()