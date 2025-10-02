import sys
from pathlib import Path

# Import pipeline function directly
from pipeline import pipeline_from_tlsf  # use the TLSF-based pipeline


def generate_traces(
    tlsf_file: str, config_file: str, num_traces: int, output_file: str
):
    pos_lines = []
    neg_lines = []

    for i in range(num_traces):
        print(f"[+] Run {i+1}/{num_traces}")
        result = pipeline_from_tlsf(tlsf_file, config_file)

        pos_trace = result["positive_trace"]
        neg_trace = result["negative_trace"]

        pos_status = "ACCEPTED ✅" if result["sys_accepts_positive"] else "REJECTED ❌"
        neg_status = "ACCEPTED ✅" if result["rej_accepts_negative"] else "REJECTED ❌"

        # Console summary
        print(f"    Positive Trace {i+1}: {pos_status}")
        print(f"    Negative Trace {i+1}: {neg_status}")

        pos_lines.append(pos_trace)
        neg_lines.append(neg_trace)

    # Write output file
    content = "\n".join(pos_lines) + "\n------\n" + "\n".join(neg_lines)
    Path(output_file).write_text(content)

    print(
        f"[+] Wrote {num_traces} positive and {num_traces} negative traces to {output_file}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: batch_traces.py <spec.tlsf> <config.toml> <N> <output.txt>")
        sys.exit(1)

    tlsf_file = sys.argv[1]
    config_file = sys.argv[2]
    num_traces = int(sys.argv[3])
    output_file = sys.argv[4]

    generate_traces(tlsf_file, config_file, num_traces, output_file)
