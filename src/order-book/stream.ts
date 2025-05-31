import { Transform } from 'stream';
import { S3 } from "aws-sdk";
import { Readable } from 'stream';
import { OrderBookTimeSeries } from "./concrete/timeseries";
import { BuyOrderBook, BuyOrderBookChange } from "./concrete/buy-order-book";
import { SellOrderBook, SellOrderBookChange } from "./concrete/sell-order-book";
import { OrderBookSnapshot } from "./main";
import * as fs from 'fs';
import path from 'path';
import { RAW_DATA_DIR } from '../constants';

class BatchTransform extends Transform {
    private buffer: any[] = [];
    private batchSize: number;

    constructor(batchSize: number) {
        super({ objectMode: true });
        this.batchSize = batchSize;
    }

    _transform(chunk: any, encoding: string, callback: (error?: Error | null, data?: any) => void) {
        this.buffer.push(chunk);

        if (this.buffer.length >= this.batchSize) {
            this.push(this.buffer);
            this.buffer = [];
        }
        callback();
    }

    _flush(callback: (error?: Error | null, data?: any) => void) {
        if (this.buffer.length > 0) {
            this.push(this.buffer);
        }
        callback();
    }
}

export const batchTransform = (batchSize: number): Transform => {
    return new BatchTransform(batchSize);
};

export const orderBookTransform = (
    buyOrderBook: BuyOrderBook,
    sellOrderBook: SellOrderBook,
    timeseries: OrderBookTimeSeries
): Transform => {
    return new Transform({
        objectMode: true,
        transform(snapshots: OrderBookSnapshot[], encoding, callback) {
            const changes = snapshots.map(snapshot => {
                const timestamp = new Date(timeseries.Date.toISOString().split('T')[0] + "T" + snapshot.time_exchange.slice(0, -4) + "Z");
                return {
                    timestamp,
                    change: snapshot.is_buy
                        ? new BuyOrderBookChange(snapshot.update_type, timestamp, snapshot.entry_px, snapshot.entry_sx)
                        : new SellOrderBookChange(snapshot.update_type, timestamp, snapshot.entry_px, snapshot.entry_sx)
                };
            });

            // Apply changes in batch
            changes.forEach(({ timestamp, change }) => {
                if (change instanceof BuyOrderBookChange) {
                    buyOrderBook.apply(change);
                    timeseries.Put(buyOrderBook, timestamp);
                } else {
                    sellOrderBook.apply(change);
                    timeseries.Put(sellOrderBook, timestamp);
                }
            });

            callback(null, changes[changes.length - 1].timestamp);
        }
    });
};

async function retryOperation<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    delay: number = 1000
): Promise<T> {
    let lastError: Error;
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await operation();
        } catch (error: any) {
            lastError = error;
            if (error.code === 'XMLParserError' || error.statusCode === 503) {
                console.log(`Retry attempt ${i + 1}/${maxRetries} after error:`, error.message);
                await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
                continue;
            }
            throw error;
        }
    }
    throw lastError!;
}

export async function createS3ReadStream(
    s3Client: S3,
    bucket: string,
    key: string
): Promise<Readable> {
    const params = {
        Bucket: bucket,
        Key: key,
    };
    
    console.log(`Attempting to read file: ${key}`);
    // Get the stream directly from S3 with retry
    return retryOperation(async () => s3Client.getObject(params).createReadStream());
}

export function createLocalReadStream(filename: string): fs.ReadStream {
    return fs.createReadStream(path.join(RAW_DATA_DIR, filename))
}