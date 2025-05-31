import path from "path";
import * as fs from "fs";
import { RAW_DATA_DIR } from "../constants";
import { S3 } from "aws-sdk";

/**
 * Assumes files are named for its orderbook span date.
 * eg. 2025-02-01.csv.gz 
 * @returns Array of objects containing filename and parsed date from filename
 * Only returns files that fall within the specified date range (startDate to endDate inclusive)
 */
async function listLocalFiles(
    startDate: Date,
    endDate: Date
): Promise<{
    name: string,
    date: Date
}[]> {
    const files = fs.readdirSync(RAW_DATA_DIR);
    return files
        .map(filename => ({
            name: filename,
            date: getDateFromFilename(filename)
        }))
        .filter(file => file.date >= startDate && file.date <= endDate);

    function getDateFromFilename(filename: string): Date {
        const basename = path.basename(filename, '.csv.gz');
        return new Date(basename);
    };
}

// Option 1: Union Types (Cleaner approach)
type LocalFileOptions = {
    startDate: Date;
    endDate: Date;
    source: 'local';
};

type S3FileOptions = {
    startDate: Date;
    endDate: Date;
    source: 's3';
    s3Client: S3;
    bucket: string;
    filePattern?: string;
};

export async function listFiles(
    options: LocalFileOptions | S3FileOptions
): Promise<{
    name: string,
    date: Date
}[]> {
    if (options.source === 'local') {
        return await listLocalFiles(options.startDate, options.endDate);
    }

    // S3 logic
    const { startDate, endDate, s3Client, bucket, filePattern = "IDDI-138123+SC-BINANCE_SPOT_BTC_USDT+S-BTCUSDT.csv.gz" } = options;
    
    const filesToCheck: { path: string, date: Date }[] = [];
    let currentDate = new Date(startDate);

    while (currentDate <= endDate) {
        const dateStr = currentDate.toISOString().split('T')[0].replace(/-/g, '');
        const prefix = `T-LIMITBOOK_FULL/D-${dateStr}/E-BINANCE/`;
        const fullPath = prefix + filePattern;
        filesToCheck.push({
            path: fullPath,
            date: new Date(currentDate)
        });
        currentDate = new Date(currentDate.getTime() + 24 * 60 * 60 * 1000);
    }

    return filesToCheck.map(file => ({ name: file.path, date: file.date })).sort((a, b) => a.date.valueOf() - b.date.valueOf());
}