import sys
from processes.streaming import StreamingPipeline

def main():
    if len(sys.argv) != 4:
        print("Usage: start_streaming.py <input_kafka_topic> <output_kafka_topic> <kafka_bootstrap_servers>")
        sys.exit(1)

    input_kafka_topic = sys.argv[1]
    output_kafka_topic = sys.argv[2]
    kafka_bootstrap_servers = sys.argv[3]

    StreamingPipeline.start_streaming(input_kafka_topic, output_kafka_topic, kafka_bootstrap_servers)

if __name__ == "__main__":
    main()