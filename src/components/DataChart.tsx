"use client";

import { HousingData, WageData, cleanNumericValue, formatCurrency } from '@/lib/dataUtils';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';

interface DataChartProps {
  data: HousingData[] | WageData[];
  datasetName: string;
  datasetType: 'housing' | 'wage';
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export default function DataChart({ data, datasetName, datasetType }: DataChartProps) {
  // Process data for charts based on dataset type
  const processDataForCharts = () => {
    if (datasetType === 'housing') {
      // For housing data, look for affordability ratios or price data
      return data.slice(0, 20).map((item, index) => {
        const keys = Object.keys(item);
        const values = Object.values(item);

        // Try to find numeric columns
        const numericData: { [key: string]: number } = {};
        keys.forEach((key, idx) => {
          const numValue = cleanNumericValue(values[idx]);
          if (numValue !== null && !isNaN(numValue)) {
            numericData[key] = numValue;
          }
        });

        return {
          name: item[keys[0]] || `Item ${index + 1}`,
          ...numericData
        };
      });
    } else {
      // For wage data, process similarly
      return data.slice(0, 20).map((item, index) => {
        const keys = Object.keys(item);
        const values = Object.values(item);

        const numericData: { [key: string]: number } = {};
        keys.forEach((key, idx) => {
          const numValue = cleanNumericValue(values[idx]);
          if (numValue !== null && !isNaN(numValue)) {
            numericData[key] = numValue;
          }
        });

        return {
          name: item[keys[0]] || `Item ${index + 1}`,
          ...numericData
        };
      });
    }
  };

  const chartData = processDataForCharts();

  // Get numeric columns for chart selection
  const numericColumns = chartData.length > 0
    ? Object.keys(chartData[0]).filter(key => key !== 'name' && typeof chartData[0][key] === 'number')
    : [];

  if (chartData.length === 0 || numericColumns.length === 0) {
    return (
      <div className="p-8 text-center">
        <p className="text-gray-500">No numeric data available for visualization.</p>
        <p className="text-sm text-gray-400 mt-2">
          The dataset may contain non-numeric data or be empty.
        </p>
      </div>
    );
  }

  return (
    <div className="p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-6">
        {datasetName} - Data Visualization
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Bar Chart */}
        <div className="bg-white p-4 rounded-lg border">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Bar Chart</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="name"
                angle={-45}
                textAnchor="end"
                height={80}
                interval={0}
              />
              <YAxis tickFormatter={(value) => formatCurrency(value)} />
              <Tooltip
                formatter={(value: number) => [formatCurrency(value), '']}
                labelStyle={{ color: '#000' }}
              />
              <Legend />
              {numericColumns.slice(0, 3).map((column, index) => (
                <Bar
                  key={column}
                  dataKey={column}
                  fill={COLORS[index % COLORS.length]}
                  name={column}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Line Chart */}
        <div className="bg-white p-4 rounded-lg border">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Line Chart</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="name"
                angle={-45}
                textAnchor="end"
                height={80}
                interval={0}
              />
              <YAxis tickFormatter={(value) => formatCurrency(value)} />
              <Tooltip
                formatter={(value: number) => [formatCurrency(value), '']}
                labelStyle={{ color: '#000' }}
              />
              <Legend />
              {numericColumns.slice(0, 3).map((column, index) => (
                <Line
                  key={column}
                  type="monotone"
                  dataKey={column}
                  stroke={COLORS[index % COLORS.length]}
                  strokeWidth={2}
                  name={column}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Pie Chart - First numeric column */}
        {numericColumns.length > 0 && (
          <div className="bg-white p-4 rounded-lg border lg:col-span-2">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Distribution - {numericColumns[0]}
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={chartData.slice(0, 8)} // Limit to 8 slices for readability
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey={numericColumns[0]}
                >
                  {chartData.slice(0, 8).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => [formatCurrency(value), numericColumns[0]]} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Data Summary */}
      <div className="mt-8 bg-white p-6 rounded-lg border">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Data Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{data.length}</div>
            <div className="text-sm text-gray-600">Total Records</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{numericColumns.length}</div>
            <div className="text-sm text-gray-600">Numeric Columns</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {chartData.length}
            </div>
            <div className="text-sm text-gray-600">Chart Data Points</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">
              {datasetType === 'housing' ? 'Housing' : 'Wage'}
            </div>
            <div className="text-sm text-gray-600">Data Type</div>
          </div>
        </div>
      </div>
    </div>
  );
}
