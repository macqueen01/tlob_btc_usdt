import { parse } from "csv-parse";
import { OrderBookTimeSeries } from "./concrete/timeseries";
import { BuyOrderBook } from "./concrete/buy-order-book";
import { SellOrderBook } from "./concrete/sell-order-book";
import { createGunzip, ZlibOptions } from 'zlib';
import { batchTransform, orderBookTransform, createS3ReadStream, createLocalReadStream } from "./stream";
import { listFiles } from "./utils";
import { Endpoint, S3 } from "aws-sdk";
import { PassThrough, Transform } from 'stream';
import { S3_REGION, S3_ENDPOINT, ACCESS_KEY, SECRET_KEY, BUCKET, TRAINING_LIST } from "../coin-api/constants";
import { RAW_DATA_DIR } from "../constants";

import * as fs from "fs";
import * as path from "path";

const S3_CLIENT = new S3({
    endpoint: new Endpoint(S3_ENDPOINT),
    s3ForcePathStyle: true,
    region: S3_REGION,
    credentials: {
        accessKeyId: ACCESS_KEY,
        secretAccessKey: SECRET_KEY
    },
});

export interface OrderBookSnapshot {
    is_buy: 0 | 1;
    time_exchange: string;
    update_type: "SNAPSHOT" | "SET";
    entry_px: number;
    entry_sx: number;
}

async function processFileWithRetry(
    file: { name: string, date: Date },
    isRemote: boolean = true,
    maxRetries: number = 3,
    delayMs: number = 1000
): Promise<void> {
    let lastError: Error | undefined;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            await processFile(file, isRemote);
            return; // Success, exit retry loop
        } catch (error: any) {
            lastError = error;
            console.log(`Attempt ${attempt}/${maxRetries} failed for file ${file.name}. Error: ${error.message}`);

            if (attempt < maxRetries) {
                const delay = delayMs * attempt; // Exponential backoff
                console.log(`Waiting ${delay}ms before retry...`);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }

    console.error(`All ${maxRetries} attempts failed for file ${file.name}. Skipping file.`);
    throw lastError!;
}

function createUnzipTransform(
    bufferSize: number,
    onError?: (error: Error) => any,
    gunzipOpt?: ZlibOptions
): { bufferStream: PassThrough; gunzipStream: Transform } {
    const gunzipStream = createGunzip(gunzipOpt ?? {
        chunkSize: 1024 * 1024, // Larger chunk size for better performance
    });
    gunzipStream.on('error', onError ?? ((err: Error) => {
        console.error('Gunzip error:', err);
    }));

    const bufferStream = new PassThrough({
        highWaterMark: bufferSize, 
        objectMode: false
    });
    bufferStream.on("error", onError ?? ((err: Error) => {
        console.error('Buffer stream error:', err);
    }));

    return { bufferStream, gunzipStream };
}

async function processFile(
    file: { name: string, date: Date },
    isRemote: boolean = true
): Promise<void> {
    console.log(`Processing file: ${file.name} for date: ${file.date.toISOString()}`);
    const timeseries = new OrderBookTimeSeries(file.date);
    const buyOrderBook = new BuyOrderBook();
    const sellOrderBook = new SellOrderBook();

    try {
        const fileStream = isRemote ?
            await createS3ReadStream(S3_CLIENT, BUCKET, file.name) :
            createLocalReadStream(file.name)

        await new Promise((resolve, reject) => {
            const errorHandler = (error: Error) => {
                console.error(`Stream error for file ${file.name}:`, {
                    fileName: file.name,
                    fileDate: file.date.toISOString(),
                    errorName: error.name,
                    errorMessage: error.message,
                    errorCode: (error as any).code,
                    errno: (error as any).errno
                });
                reject(error);
            };

            let pipeline = fileStream;
            const { bufferStream, gunzipStream } = createUnzipTransform(
                1024 * 1024 * 1, // 1MB buffer
                errorHandler
            );

            pipeline
                .pipe(bufferStream)
                .pipe(gunzipStream)
                .pipe(parse({
                    columns: true,
                    skip_empty_lines: true,
                    delimiter: ';',
                    quote: '"',
                    escape: '\\',
                    cast: true,
                    cast_date: true,
                    relax_column_count: true, // Allow inconsistent column counts
                    skip_records_with_error: true // Skip malformed records instead of failing
                }))
                .pipe(batchTransform(1000))
                .pipe(orderBookTransform(buyOrderBook, sellOrderBook, timeseries))
                .on("data", (timestamp: Date) => {
                    console.log(`Processed batch up to: ${timestamp.toISOString()}`);
                })
                .on("end", () => {
                    console.log(`Completed processing file: ${file.name}`);
                    resolve(undefined);
                })
                .on("error", (error: Error) => {
                    console.error(`Error processing file ${file.name}:`, error);
                    console.error(`Error details:`, {
                        name: error.name,
                        message: error.message,
                        stack: error.stack
                    });
                    reject(error);
                });
        });
    } catch (error: any) {
        console.error(`Failed to process file ${file.name}:`, error);
        console.error(`Error details:`, {
            name: error.name,
            message: error.message,
            stack: error.stack
        });
        throw error; // Re-throw for retry logic
    } finally {
        timeseries.close();
    }
}

async function processFilesInParallel(
    files: { name: string, date: Date }[],
    isRemote: boolean = true,
    concurrency: number = 3
): Promise<void> {
    const chunks: { name: string, date: Date }[][] = [];
    for (let i = 0; i < files.length; i += concurrency) {
        chunks.push(files.slice(i, i + concurrency));
    }

    for (const chunk of chunks) {
        await Promise.all(
            chunk.map(file => processFileWithRetry(file, isRemote).catch(error => {
                console.error(`Permanently failed to process file ${file.name}, continuing with other files...`);
            }))
        );
    }
}

export async function main(downloadOnly: boolean, from: string, to: string, remote: boolean = true, numParallel: number = 10) {
    if (downloadOnly) return await downloadFromS3AndUnzip()

    const startDate = new Date(from);
    const endDate = new Date(to);

    try {
        const files = remote ?
            await listFiles({
                source: 's3',
                startDate,
                endDate,
                s3Client: S3_CLIENT,
                bucket: BUCKET
            }) :
            await listFiles({
                source: "local",
                startDate,
                endDate
            });
        console.log(`Found ${files.length} files to process`);

        await processFilesInParallel(files, remote, numParallel);
    } catch (error) {
        console.error("Error listing files:", error);
    }
};

async function downloadFromS3AndUnzip(): Promise<void> {
    // Ensure download directory exists
    if (!fs.existsSync(RAW_DATA_DIR)) {
        fs.mkdirSync(RAW_DATA_DIR, { recursive: true });
    }

    // Get list of files to download from training_list dates
    const files: { name: string, date: Date }[] = [];
    
    for (const date of TRAINING_LIST) {
        const dateStr = date.toISOString().split('T')[0].replace(/-/g, '');
        const prefix = `T-LIMITBOOK_FULL/D-${dateStr}/E-BINANCE/`;
        const filePattern = "IDDI-138123+SC-BINANCE_SPOT_BTC_USDT+S-BTCUSDT.csv.gz";
        const fullPath = prefix + filePattern;
        files.push({
            name: fullPath,
            date: new Date(date)
        });
    }

    console.log(`Found ${files.length} files to download and unzip`);

    for (const file of files) {
        try {
            console.log(`Downloading and unzipping: ${file.name}`);
            
            // Create S3 read stream
            const s3ReadStream = await createS3ReadStream(S3_CLIENT, BUCKET, file.name);
            
            // Create unzip transform
            const unzipStream = createGunzip();
            
            // Generate ISO date string filename
            const isoDateString = file.date.toISOString().split('T')[0]; // YYYY-MM-DD format
            const outputFilename = `${isoDateString}.csv`;
            const outputPath = path.join(RAW_DATA_DIR, outputFilename);
            
            // Create write stream
            const writeStream = fs.createWriteStream(outputPath);
            
            // Pipeline: S3 → Unzip → File
            await new Promise<void>((resolve, reject) => {
                s3ReadStream
                    .pipe(unzipStream)
                    .pipe(writeStream)
                    .on('finish', () => {
                        console.log(`Successfully saved: ${outputFilename}`);
                        resolve();
                    })
                    .on('error', reject);
            });
            
        } catch (error) {
            console.error(`Failed to download/unzip ${file.name}:`, error);
        }
    }
}