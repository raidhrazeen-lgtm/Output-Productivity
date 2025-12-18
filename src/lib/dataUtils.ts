import * as XLSX from 'xlsx';
import Papa from 'papaparse';

// Types for our data
export interface HousingData {
  [key: string]: any;
}

export interface WageData {
  [key: string]: any;
}

export interface DatasetInfo {
  name: string;
  path: string;
  type: 'housing' | 'wage';
  description: string;
}

// List of available datasets
export const AVAILABLE_DATASETS: DatasetInfo[] = [
  {
    name: 'Housing Purchase Affordability UK',
    path: '/data/housingpurchaseaffordabilityukbycountryandenglishregion2024.xlsx',
    type: 'housing',
    description: 'UK housing affordability data by country and English region for 2024'
  },
  {
    name: 'Annual Pay - Incentive 2025',
    path: '/data/PROV - Work Region PubPriv Table 25.8a   Annual pay - Incentive 2025.xlsx',
    type: 'wage',
    description: 'Annual pay including incentives for 2025'
  },
  {
    name: 'Weekly Pay - Gross 2025',
    path: '/data/PROV - Work Region PubPriv Table 25.1a   Weekly pay - Gross 2025.xlsx',
    type: 'wage',
    description: 'Weekly gross pay data for 2025'
  },
  {
    name: 'Hourly Pay - Gross 2025',
    path: '/data/PROV - Work Region PubPriv Table 25.5a   Hourly pay - Gross 2025.xlsx',
    type: 'wage',
    description: 'Hourly gross pay data for 2025'
  },
  {
    name: 'Paid Hours Worked - Total 2025',
    path: '/data/PROV - Work Region PubPriv Table 25.9a   Paid hours worked - Total 2025.xlsx',
    type: 'wage',
    description: 'Total paid hours worked for 2025'
  }
];

// Function to read Excel file and return JSON data
export async function readExcelFile(filePath: string): Promise<any[]> {
  try {
    const response = await fetch(filePath);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const arrayBuffer = await response.arrayBuffer();
    const workbook = XLSX.read(arrayBuffer, { type: 'array' });

    // Get the first worksheet
    const sheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[sheetName];

    // Convert to JSON
    const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

    return jsonData;
  } catch (error) {
    console.error('Error reading Excel file:', error);
    throw new Error(`Failed to read Excel file: ${filePath}`);
  }
}

// Function to process housing affordability data
export function processHousingData(rawData: any[]): HousingData[] {
  if (!rawData || rawData.length < 2) return [];

  // Assume first row is headers
  const headers = rawData[0];
  const data = rawData.slice(1);

  return data.map(row => {
    const obj: HousingData = {};
    headers.forEach((header: string, index: number) => {
      obj[header] = row[index];
    });
    return obj;
  }).filter(item => item && Object.keys(item).length > 0);
}

// Function to process wage data
export function processWageData(rawData: any[]): WageData[] {
  if (!rawData || rawData.length < 2) return [];

  // Assume first row is headers
  const headers = rawData[0];
  const data = rawData.slice(1);

  return data.map(row => {
    const obj: WageData = {};
    headers.forEach((header: string, index: number) => {
      obj[header] = row[index];
    });
    return obj;
  }).filter(item => item && Object.keys(item).length > 0);
}

// Function to get dataset by type
export async function getDatasetData(dataset: DatasetInfo): Promise<HousingData[] | WageData[]> {
  const rawData = await readExcelFile(dataset.path);

  if (dataset.type === 'housing') {
    return processHousingData(rawData);
  } else {
    return processWageData(rawData);
  }
}

// Function to get all datasets data
export async function getAllDatasetsData(): Promise<{
  [key: string]: HousingData[] | WageData[]
}> {
  const results: { [key: string]: HousingData[] | WageData[] } = {};

  for (const dataset of AVAILABLE_DATASETS) {
    try {
      results[dataset.name] = await getDatasetData(dataset);
    } catch (error) {
      console.error(`Failed to load dataset ${dataset.name}:`, error);
      results[dataset.name] = [];
    }
  }

  return results;
}

// Utility function to clean numeric values
export function cleanNumericValue(value: any): number | null {
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const cleaned = value.replace(/[£,%]/g, '').trim();
    const num = parseFloat(cleaned);
    return isNaN(num) ? null : num;
  }
  return null;
}

// Utility function to format currency
export function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-GB', {
    style: 'currency',
    currency: 'GBP'
  }).format(value);
}

// Utility function to format percentage
export function formatPercentage(value: number): string {
  return `${value.toFixed(2)}%`;
}
