import pika
from idbscan import IncrementalDBSCAN

dbscan = IncrementalDBSCAN()
batch = 0

def send_to_incremental_dbscan(message):
    dbscan.set_data(message)
    # This is going to be done only the first few times until the DBSCAN creates the first clusters.
    # The initial data will be used to feed the DBSCAN in order to create the first clusters and then
    # the Incremental DBSCAN will be used.
    if batch < 500:
        dbscan.batch_dbscan()
    if batch >= 500:
        dbscan.incremental_dbscan_()


def callback(ch, method, properties, body):
    print("[x] Received %r" % body.decode())
    send_to_incremental_dbscan(body.decode())
    global batch
    batch += 1


def main():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host='localhost'
        )
    )
    channel = connection.channel()
    channel.queue_declare(queue='hell')
    first_row = True
    channel.basic_consume('hell', callback, auto_ack=True)
    print("[x] Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()

if __name__ == '__main__':
    main()
