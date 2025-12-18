"use client";

import { useState, useEffect } from 'react';
import { AVAILABLE_DATASETS, getAllDatasetsData, HousingData, WageData } from '@/lib/dataUtils';
import DataChart from './DataChart';
import DataTable from './DataTable';

export default function Dashboard() {
  const [datasets, setDatasets] = useState<{ [key: string]: HousingData[] | WageData[] }>({});
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'charts' | 'table'>('charts');

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getAllDatasetsData();
      setDatasets(data);
      // Set first dataset as default
      const firstDatasetName = Object.keys(data)[0];
      if (firstDatasetName) {
        setSelectedDataset(firstDatasetName);
      }
    } catch (err) {
      setError('Failed to load datasets. Please check file paths and permissions.');
      console.error('Error loading datasets:', err);
    } finally {
      setLoading(false);
    }
  };

  const currentDataset = datasets[selectedDataset];
  const currentDatasetInfo = AVAILABLE_DATASETS.find(d => d.name === selectedDataset);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading datasets...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center max-w-md">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-xl font-semibold text-gray-800 mb-2">Error Loading Data</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={loadDatasets}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Data Dashboard</h1>
              <p className="mt-1 text-sm text-gray-500">
                UK Housing and Wage Data Analysis
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-500">
                {AVAILABLE_DATASETS.length} datasets loaded
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Dataset Selector */}
        <div className="mb-8">
          <label htmlFor="dataset-select" className="block text-sm font-medium text-gray-700 mb-2">
            Select Dataset
          </label>
          <select
            id="dataset-select"
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="block w-full max-w-md px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          >
            {AVAILABLE_DATASETS.map((dataset) => (
              <option key={dataset.name} value={dataset.name}>
                {dataset.name}
              </option>
            ))}
          </select>
          {currentDatasetInfo && (
            <p className="mt-2 text-sm text-gray-600">{currentDatasetInfo.description}</p>
          )}
        </div>

        {/* Tab Navigation */}
        <div className="mb-6">
          <nav className="flex space-x-8">
            <button
              onClick={() => setActiveTab('charts')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'charts'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Charts & Visualizations
            </button>
            <button
              onClick={() => setActiveTab('table')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'table'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Data Table
            </button>
          </nav>
        </div>

        {/* Content Area */}
        <div className="bg-white rounded-lg shadow-sm border">
          {currentDataset && currentDataset.length > 0 ? (
            activeTab === 'charts' ? (
              <DataChart
                data={currentDataset}
                datasetName={selectedDataset}
                datasetType={currentDatasetInfo?.type || 'housing'}
              />
            ) : (
              <DataTable
                data={currentDataset}
                datasetName={selectedDataset}
              />
            )
          ) : (
            <div className="p-8 text-center">
              <p className="text-gray-500">No data available for the selected dataset.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
