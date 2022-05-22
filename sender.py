from time import sleep
import pika


def main():
    #rabbitmq_ip = open('../rabbitmq/rabbitmq_ip', 'r')
    connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                    host='localhost'
                )
        )
    channel = connection.channel()

    channel.queue_declare(queue='hellow')
    try:
        with open('data/test_file.csv') as fp:
            for line in fp:
                message = line
                channel.basic_publish(exchange='',
                                      routing_key='hellow',
                                      body=message)

                print('[*] Now published: ' + message)

                # Publish a row to RabbitMQ every 3 seconds
                sleep(3)

        connection.close()
    except TimeoutError:
        print('Timeout exception: Connection with RabbitMQ failed. ', TimeoutError)

if __name__ == '__main__':
    main()
