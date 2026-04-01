import json
import random
import time
import argparse
import logging
import requests
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

KAFKA_TOPIC   = "fraud-detection-transactions"
KAFKA_BROKER  = "localhost:9092"
API_URL       = "http://localhost:8000/predict"
FRAUD_RATE    = 0.02  # 2% fraud rate


def generate_transaction(is_fraud: bool = False) -> dict:
    base_amount = random.uniform(500, 5000) if is_fraud else random.uniform(1, 500)
    return {
        "transaction_id": f"TID_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
        "timestamp":      datetime.utcnow().isoformat(),
        "is_fraud_label": is_fraud,
        "features": {
            "Time":   random.uniform(0, 172792),
            "Amount": base_amount,
            **{f"V{i}": np.random.normal(-2 if is_fraud else 0, 1.5) for i in range(1, 29)}
        }
    }


def run_producer(rate_per_second: int = 5):
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        logger.info(f"Producer started — sending {rate_per_second} transactions/second to '{KAFKA_TOPIC}'")

        count = 0
        while True:
            is_fraud = random.random() < FRAUD_RATE
            tid      = generate_transaction(is_fraud=is_fraud)
            producer.send(KAFKA_TOPIC, value=tid)
            count += 1
            if count % 50 == 0:
                logger.info(f"Sent {count} transactions")
            time.sleep(1 / rate_per_second)

    except ImportError:
        logger.error("kafka-python not installed")
    except Exception as e:
        logger.error(f"Producer error: {e}")


def run_consumer():
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="fraud-detection-group",
            auto_offset_reset="latest"
        )
        logger.info(f"Consumer started — listening on '{KAFKA_TOPIC}'")

        fraud_count = 0
        total_count = 0

        for message in consumer:
            tid      = message.value
            features = tid["features"]
            total_count += 1

            try:
                response = requests.post(API_URL, json=features, timeout=1)
                result   = response.json()

                if result["is_fraud"]:
                    fraud_count += 1
                    logger.warning(
                        f"FRAUD DETECTED | TID: {tid['transaction_id']} | "
                        f"Amount: {features['Amount']:.2f} DH | "
                        f"Probability: {result['fraud_probability']:.4f} | "
                        f"Risk: {result['risk_level']} | "
                        f"Latency: {result['inference_time_ms']}ms"
                    )
                else:
                    if total_count % 100 == 0:
                        logger.info(
                            f"Processed {total_count} transactions | "
                            f"Frauds detected: {fraud_count} ({fraud_count/total_count:.2%})"
                        )

            except requests.exceptions.RequestException as e:
                logger.error(f"API call failed: {e}")

    except ImportError:
        logger.error("kafka-python not installed")
    except Exception as e:
        logger.error(f"Consumer error: {e}")


def run_simulation(n_transactions: int = 200):
    logger.info(f"Starting simulation — {n_transactions} transactions")
    logger.info("=" * 60)

    results = {"total": 0, "fraud_detected": 0, "correct": 0, "latencies": []}

    for i in range(n_transactions):
        is_fraud = random.random() < FRAUD_RATE
        tid      = generate_transaction(is_fraud=is_fraud)
        features = tid["features"]

        try:
            response = requests.post(API_URL, json=features, timeout=2)
            result   = response.json()

            results["total"] += 1
            results["latencies"].append(result["inference_time_ms"])

            if result["is_fraud"]:
                results["fraud_detected"] += 1

            if result["is_fraud"] == is_fraud:
                results["correct"] += 1

            if result["is_fraud"]:
                logger.warning(
                    f"🚨 [{i+1}/{n_transactions}] FRAUD | "
                    f"Amount: {features['Amount']:.2f} | "
                    f"Prob: {result['fraud_probability']:.4f} | "
                    f"Latency: {result['inference_time_ms']}ms"
                )

        except requests.exceptions.RequestException:
            logger.error("API not reachable")
            break

        # 50ms between transactions
        time.sleep(0.05)  

    if results["total"] > 0:
        logger.info("\n" + "=" * 60)
        logger.info("SIMULATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total transactions processed : {results['total']}")
        logger.info(f"Frauds detected              : {results['fraud_detected']}")
        logger.info(f"Detection rate               : {results['fraud_detected']/results['total']:.2%}")
        logger.info(f"Avg inference latency        : {np.mean(results['latencies']):.2f}ms")
        logger.info(f"P99 inference latency        : {np.percentile(results['latencies'], 99):.2f}ms")
        logger.info("=" * 60)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka Fraud Detection Simulator")
    parser.add_argument("--mode", choices=["producer", "consumer", "simulate"], default="simulate")
    parser.add_argument("--n",    type=int, default=200, help="Number of transactions for simulation mode")
    parser.add_argument("--rate", type=int, default=5,   help="Transactions per second for producer mode")
    args = parser.parse_args()

    if args.mode == "producer":
        run_producer(rate_per_second=args.rate)
    elif args.mode == "consumer":
        run_consumer()
    else:
        run_simulation(n_transactions=args.n)
