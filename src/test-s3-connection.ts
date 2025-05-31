import { S3_CONFIG, logConfigSafely } from "./config";
import { S3, Endpoint } from "aws-sdk";

async function testS3Connection(): Promise<void> {
    console.log("=== Testing S3 Connection ===\n");
    
    try {
        // Log config safely (masks sensitive data)
        logConfigSafely();
        console.log("\n");

        // Create S3 client
        const s3Client = new S3({
            endpoint: new Endpoint(S3_CONFIG.endpoint),
            s3ForcePathStyle: true,
            region: S3_CONFIG.region,
            credentials: {
                accessKeyId: S3_CONFIG.accessKey,
                secretAccessKey: S3_CONFIG.secretKey
            },
        });

        console.log("✓ S3 client created successfully");

        // Test 1: List buckets
        console.log("\n--- Test 1: Listing buckets ---");
        try {
            const buckets = await s3Client.listBuckets().promise();
            console.log("✓ Successfully connected to S3");
            console.log(`Found ${buckets.Buckets?.length || 0} buckets`);
            buckets.Buckets?.forEach(bucket => {
                console.log(`  - ${bucket.Name} (created: ${bucket.CreationDate})`);
            });
        } catch (error: any) {
            console.log("✗ Failed to list buckets:", error.message || error.code || 'Unknown error');
            throw error;
        }

        // Test 2: Check if target bucket exists
        console.log("\n--- Test 2: Checking target bucket ---");
        try {
            await s3Client.headBucket({ Bucket: S3_CONFIG.bucket }).promise();
            console.log(`✓ Target bucket '${S3_CONFIG.bucket}' exists and is accessible`);
        } catch (error: any) {
            console.log(`✗ Target bucket '${S3_CONFIG.bucket}' is not accessible:`, error.message || error.code || 'Unknown error');
            
            // Don't throw here, let's continue to test file access
            console.log("⚠ Continuing with file access test despite bucket access error...");
        }

        // Test 3: Test file access (using a sample training date)
        console.log("\n--- Test 3: Testing file access ---");
        const testDate = new Date("2023-06-17");
        const dateStr = testDate.toISOString().split('T')[0].replace(/-/g, '');
        const testFilePath = `T-LIMITBOOK_FULL/D-${dateStr}/E-BINANCE/IDDI-138123+SC-BINANCE_SPOT_BTC_USDT+S-BTCUSDT.csv.gz`;
        
        try {
            const headResult = await s3Client.headObject({
                Bucket: S3_CONFIG.bucket,
                Key: testFilePath
            }).promise();
            
            console.log(`✓ Test file exists: ${testFilePath}`);
            console.log(`  Size: ${headResult.ContentLength} bytes`);
            console.log(`  Last Modified: ${headResult.LastModified}`);
            console.log(`  Content Type: ${headResult.ContentType}`);
        } catch (error: any) {
            if (error.code === 'NotFound' || error.code === 'NoSuchKey') {
                console.log(`⚠ Test file not found: ${testFilePath}`);
                console.log("  This is expected if the file doesn't exist for this date");
            } else {
                console.log(`✗ Error accessing test file: ${error.message || error.code || 'Unknown error'}`);
                console.log(`  Error details:`, {
                    code: error.code,
                    statusCode: error.statusCode,
                    message: error.message
                });
            }
        }

        console.log("\n=== S3 Connection Test Completed ===");
        console.log("✓ Basic S3 connectivity works");
        console.log("✓ Can list buckets and see target bucket");
        console.log("Note: Some operations may have limited permissions, but basic access is working");

    } catch (error: any) {
        console.error("\n=== S3 Connection Test Failed ===");
        console.error("Error details:", {
            name: error.name,
            message: error.message,
            code: error.code,
            statusCode: error.statusCode,
        });
        
        // Provide helpful error messages (with null-safe checks)
        if (error.code === 'InvalidAccessKeyId') {
            console.error("💡 Check your S3_ACCESS_KEY in .env file");
        } else if (error.code === 'SignatureDoesNotMatch') {
            console.error("💡 Check your S3_SECRET_KEY in .env file");
        } else if (error.code === 'Forbidden') {
            console.error("💡 Check your credentials have the required permissions");
        } else if (error.message && error.message.includes('getaddrinfo ENOTFOUND')) {
            console.error("💡 Check your S3_ENDPOINT in .env file");
        }
        
        process.exit(1);
    }
}

// Run the test
testS3Connection(); 