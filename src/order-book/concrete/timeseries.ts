import { TimeSeries, CSVable } from "../order-book";
import { BuyOrderBook } from "./buy-order-book";
import { SellOrderBook } from "./sell-order-book";
import { Writable } from "stream";
import * as fs from 'fs';
import * as path from 'path';

export class OrderBookTimeSeries implements TimeSeries, CSVable {
    private recentTimestamp: number = 0;
    private sellOrderBook?: SellOrderBook;
    private buyOrderBook?: BuyOrderBook;
    private writableStream: Writable;
    private depth: 10 | 20 | 50 | 80 | 120;
    private aggregatingWindow: number = 200;
    private writeBuffer: string[] = [];
    private readonly WRITE_BUFFER_SIZE = 1000;

    constructor(
        private readonly targetDate: Date, 
        outputDir: string = 'preprocessed', 
        depth: 10 | 20 | 50 | 80 | 120 = 10
    ) {
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        const filePath = path.join(outputDir, `orderbook_${targetDate.toISOString().split("T")[0]}.csv`);
        this.writableStream = fs.createWriteStream(filePath);

        // Write CSV header
        const header = ['timestamp'];
        
        for (let i = 1; i <= depth; i++) {
            header.push(`sell_price_${i}`, `sell_size_${i}`);
        }
        
        for (let i = 1; i <= depth; i++) {
            header.push(`buy_price_${i}`, `buy_size_${i}`);
        }
        this.depth = depth;
        this.writableStream.write(header.join(',') + '; \n');
    }

    Put(obj: BuyOrderBook | SellOrderBook, at: Date): void {
        if (this.isNewTimeWindow(at)) {
            const row = this.toRow(this.depth).join(',');
            this.writeBuffer.push(row);
            
            if (this.writeBuffer.length >= this.WRITE_BUFFER_SIZE) {
                this.writableStream.write(this.writeBuffer.join('\n') + '\n');
                this.writeBuffer = [];
            }
        }

        this.updateOrderBook(obj);
        this.updateTimestamp(at);
    };

    private isNewTimeWindow(at: Date): boolean {
        const flooredTimestamp = this.flooredTimestamp(at);
        return this.recentTimestamp != 0 && 
            this.recentTimestamp != flooredTimestamp;
    }

    private flooredTimestamp(datetime: Date): number {
        return Math.floor(datetime.valueOf() / this.aggregatingWindow) * this.aggregatingWindow;
    }

    private updateTimestamp(datetime: Date): void {
        this.recentTimestamp = this.flooredTimestamp(datetime);
    }

    private updateOrderBook(book: BuyOrderBook | SellOrderBook): void {
        if (book instanceof BuyOrderBook) {
            this.buyOrderBook = book;  // Just reference it
            return;
        }
        this.sellOrderBook = book;
    }

    get Date(): Date {
        return this.targetDate;
    }

    toRow(depth: 10 | 20 | 50 | 80 | 120 = 10): (string | number)[] {
        const timestamp = this.recentTimestamp;
        const sellValues = this.sellOrderBook?.toRow(depth) ?? [];
        const buyValues = this.buyOrderBook?.toRow(depth) ?? [];
        return [timestamp, ...sellValues, ...buyValues];
    }

    close(): void {
        // Write any remaining buffered rows
        if (this.writeBuffer.length > 0) {
            this.writableStream.write(this.writeBuffer.join('\n') + '\n');
        }
        this.writableStream?.end();
    }
};

