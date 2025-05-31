import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Helper function to get required env variables
function getEnvVar(name: string, defaultValue?: string): string {
    const value = process.env[name] || defaultValue;
    if (!value) {
        throw new Error(`Environment variable ${name} is required`);
    }
    return value;
}

// Helper function for sensitive env variables (no defaults)
function getSecretEnvVar(name: string): string {
    const value = process.env[name];
    if (!value) {
        throw new Error(`Secret environment variable ${name} is required and must be provided via .env file or environment`);
    }
    return value;
}

// S3 Configuration
export const S3_CONFIG = {
    region: getEnvVar('S3_REGION', 'us-east-1'),
    bucket: getSecretEnvVar('S3_BUCKET'), // No default - must be provided
    accessKey: getSecretEnvVar('S3_ACCESS_KEY'), // No default - must be provided
    secretKey: getSecretEnvVar('S3_SECRET_KEY'), // No default - must be provided
    endpoint: getSecretEnvVar('S3_ENDPOINT'), // No default - must be provided
};

// Directory Configuration (safe defaults)
export const DIRS = {
    rawData: getEnvVar('RAW_DATA_DIR', './raw'),
    logs: getEnvVar('LOG_DIR', './logs'),
};

// Helper function to log config safely (masks sensitive values)
export function logConfigSafely(): void {
    console.log('Configuration loaded:');
    console.log('S3_REGION:', S3_CONFIG.region);
    console.log('S3_BUCKET:', S3_CONFIG.bucket);
    console.log('S3_ACCESS_KEY:', S3_CONFIG.accessKey.substring(0, 8) + '...[MASKED]');
    console.log('S3_SECRET_KEY:', '[MASKED]');
    console.log('S3_ENDPOINT:', S3_CONFIG.endpoint);
    console.log('RAW_DATA_DIR:', DIRS.rawData);
    console.log('LOG_DIR:', DIRS.logs);
} 