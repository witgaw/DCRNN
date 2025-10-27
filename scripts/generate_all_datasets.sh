#!/bin/bash
# Generate training data for both METR-LA and PEMS-BAY datasets

set -e  # Exit on error

echo "=========================================="
echo "Generating DCRNN Training Datasets"
echo "=========================================="
echo ""

# Generate METR-LA
echo "📊 Processing METR-LA dataset..."
echo "=========================================="
python scripts/generate_training_data.py \
    --traffic_df_filename data/metr-la.h5 \
    --output_dir data/hf_datasets \
    --format parquet

echo ""
echo "✅ METR-LA complete!"
echo ""

# Generate PEMS-BAY
echo "📊 Processing PEMS-BAY dataset..."
echo "=========================================="
python scripts/generate_training_data.py \
    --traffic_df_filename data/pems-bay.h5 \
    --output_dir data/hf_datasets \
    --format parquet

echo ""
echo "✅ PEMS-BAY complete!"
echo ""

echo "=========================================="
echo "🎉 All datasets generated successfully!"
echo "=========================================="
echo ""
echo "Output locations:"
echo "  - data/hf_datasets/METR-LA/"
echo "  - data/hf_datasets/PEMS-BAY/"
