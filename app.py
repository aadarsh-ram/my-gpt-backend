import pika
import os
import signal
import sys

from dotenv import load_dotenv
from model_inference import summarize_pdf

load_dotenv()
MQ_SEND_QUEUE = os.getenv('MQ_SEND_QUEUE')
MQ_RECV_QUEUE = os.getenv('MQ_RECV_QUEUE')
QUEUE_HEARTBEAT = os.getenv('QUEUE_HEARTBEAT')

def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', heartbeat=int(QUEUE_HEARTBEAT)))
    channel = connection.channel()

    def summarization_callback(ch, method, properties, body):
        print (f"Received file path {body}")
        result = summarize_pdf(body.decode())
        channel.basic_publish(exchange='', routing_key=MQ_SEND_QUEUE, body=result)
        print ('[*] Sent summary!')

    # Declare queues
    channel.queue_declare(queue=MQ_RECV_QUEUE, durable=True)
    channel.queue_declare(queue=MQ_SEND_QUEUE, durable=True)

    # Consumer process
    channel.basic_consume(queue=MQ_RECV_QUEUE, on_message_callback=summarization_callback, auto_ack=True)

    print('[*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print ("Closing...")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)