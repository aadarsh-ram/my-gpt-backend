import pika
import os
import json
import sys

from dotenv import load_dotenv
from llm_model_inference import summarize_pdf, grammar_check

load_dotenv()
MQ_SEND_QUEUE = os.getenv('MQ_SEND_QUEUE')
MQ_RECV_QUEUE = os.getenv('MQ_RECV_QUEUE')
QUEUE_HEARTBEAT = os.getenv('QUEUE_HEARTBEAT')

def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', heartbeat=int(QUEUE_HEARTBEAT)))
    channel = connection.channel()

    # Main callback
    def router_callback(ch, method, properties, body):
        print (f"Received message from client {body}")
        client_json = json.loads(body.decode())
        result_json = {}

        if client_json['type'] == 'summary':
            result = summarize_pdf(client_json['content'])
            if isinstance(result, Exception):
                result_json['status'] = 'fail'
                result_json['content'] = result.args[0]
            else:
                result_json['status'] = 'success'
                result_json['content'] = result
            
            print ('[*] Sent summary!')
        
        elif client_json['type'] == 'grammar':
            result = grammar_check(client_json['content'])
            if isinstance(result, Exception):
                result_json['status'] = 'fail'
                result_json['content'] = result.args[0]
            else:
                result_json['status'] = 'success'
                result_json['content'] = result
            print ('[*] Sent corrected text!')
        
        channel.basic_publish(exchange='', routing_key=MQ_SEND_QUEUE, body=json.dumps(result_json))

    # Declare queues
    channel.queue_declare(queue=MQ_RECV_QUEUE, durable=True)
    channel.queue_declare(queue=MQ_SEND_QUEUE, durable=True)

    # Consumer process
    channel.basic_consume(queue=MQ_RECV_QUEUE, on_message_callback=router_callback, auto_ack=True)

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