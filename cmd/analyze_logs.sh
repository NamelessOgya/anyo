#!/usr/bin/env bash
C=${C:-ilora-dllm2rec-dev}

while getopts ":c:" opt; do
  case $opt in
    c) C="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

shift $((OPTIND-1))

echo "Targeting Container: $C"
docker exec "$C" bash -c "PYTHONPATH=/workspace /opt/conda/bin/poetry run python src/utils/analyze_tb_logs.py $@"
