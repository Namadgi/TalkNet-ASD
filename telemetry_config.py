"""
OpenTelemetry Configuration for Different Environments

This file shows how to configure OpenTelemetry for different deployment scenarios:
- Local development (console output)
- GCP production (Cloud Trace + Cloud Monitoring)
- Other cloud providers (AWS X-Ray, Azure Monitor)
"""

import os
from typing import Optional

def setup_telemetry_for_environment():
    """
    Configure OpenTelemetry for GCP deployment.

    Only GCP telemetry is supported. Make sure to set:
    - GOOGLE_CLOUD_PROJECT environment variable
    - Appropriate GCP authentication (automatic in Cloud Run)
    """

    env = os.getenv('OTEL_ENV', 'gcp').lower()

    if env == 'gcp':
        setup_gcp_telemetry()
    else:
        print(f"Warning: Only GCP telemetry is supported. Using GCP setup for env: {env}")
        setup_gcp_telemetry()

def setup_gcp_telemetry():
    """Google Cloud Platform - Cloud Trace + Cloud Monitoring"""
    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
        from opentelemetry.sdk.resources import Resource
        import uuid

        # Differentiate instances and workers using Instance ID and PID
        instance_id = os.getenv("CLOUD_RUN_INSTANCE", os.getenv("HOSTNAME", str(uuid.uuid4())))
        process_id = os.getpid()
        host_id = f"{instance_id}-{process_id}"

        resource = Resource.create({
            "service.name": os.getenv("OTEL_SERVICE_NAME", "ml-asd"),
            "host.id": host_id
        })

        # Tracing
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer_provider = trace.get_tracer_provider()
        tracer_provider.add_span_processor(
            BatchSpanProcessor(CloudTraceSpanExporter())
        )

        # Metrics
        metrics.set_meter_provider(MeterProvider(
            metric_readers=[PeriodicExportingMetricReader(CloudMonitoringMetricsExporter())],
            resource=resource
        ))

        print("✅ OpenTelemetry configured for GCP (Cloud Trace + Cloud Monitoring)")

    except ImportError as e:
        print(f"❌ GCP telemetry setup failed: {e}")
        print("Install required packages: pip install opentelemetry-exporter-gcp-trace opentelemetry-exporter-gcp-monitoring")
        raise e

# GCP Configuration
TELEMETRY_CONFIGS = {
    'gcp': {
        'description': 'Google Cloud Platform with Cloud Trace and Cloud Monitoring',
        'env_vars': {
            'GOOGLE_CLOUD_PROJECT': 'biometry-416410',
            'GOOGLE_APPLICATION_CREDENTIALS': 'gcp.json'
        },
        'requirements': [
            'opentelemetry-exporter-gcp-trace',
            'opentelemetry-exporter-gcp-monitoring',
            'google-cloud-trace',
            'google-cloud-monitoring'
        ]
    },

    'gcp_cloud_run': {
        'description': 'GCP Cloud Run with automatic authentication (recommended)',
        'env_vars': {
            'GOOGLE_CLOUD_PROJECT': 'biometry-416410'
            # No need for GOOGLE_APPLICATION_CREDENTIALS in Cloud Run
        },
        'requirements': [
            'opentelemetry-exporter-gcp-trace',
            'opentelemetry-exporter-gcp-monitoring',
            'google-cloud-trace',
            'google-cloud-monitoring'
        ]
    }
}

if __name__ == "__main__":
    print("🔧 GCP OpenTelemetry Configuration")
    print("=" * 40)

    current_env = os.getenv('OTEL_ENV', 'gcp')
    print(f"Current environment: {current_env.upper()}")

    if current_env in TELEMETRY_CONFIGS:
        config = TELEMETRY_CONFIGS[current_env]
        print(f"Description: {config['description']}")
        print("\n📋 Required environment variables:")
        for key, value in config['env_vars'].items():
            if isinstance(value, str):
                print(f"  {key}={value}")
            else:
                print(f"  {key}=[automatically set by GCP]")

        print("\n📦 Required packages:")
        for pkg in config['requirements']:
            print(f"  pip install {pkg}")

    print("\n🚀 GCP Deployment:")
    print("  export OTEL_ENV=gcp")
    print("  export GOOGLE_CLOUD_PROJECT=biometry-416410")
    print("  # For Cloud Run: no GOOGLE_APPLICATION_CREDENTIALS needed")
    print("  # For GKE/other: set GOOGLE_APPLICATION_CREDENTIALS if needed")

    print("\n📊 Telemetry Access:")
    print("  Cloud Trace: https://console.cloud.google.com/traces")
    print("  Cloud Monitoring: https://console.cloud.google.com/monitoring")
    print("  Cloud Logging: https://console.cloud.google.com/logs")


