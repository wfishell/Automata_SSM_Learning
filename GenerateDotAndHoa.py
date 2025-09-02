#!/usr/bin/env python3
import argparse
import subprocess

def main():
    p = argparse.ArgumentParser(description="Generate controller.dot and controller.hoa via ltlsynt")
    p.add_argument("--inputs", required=True, help="Comma-separated input APs, e.g. a,b")
    p.add_argument("--outputs", required=True, help="Comma-separated output APs, e.g. p0,p1")
    p.add_argument("--formula", required=True, help="LTL formula (quote it!)")
    p.add_argument("--dot", default="controller.dot", help="Output DOT path (default: controller.dot)")
    p.add_argument("--hoa", default="controller.hoa", help="Output HOA path (default: controller.hoa)")
    args = p.parse_args()

    # Run ltlsynt once per format, capturing stdout directly to files (no shell redirection)
    with open(args.dot, "w") as f_dot:
        subprocess.run(
            ["ltlsynt", f"--ins={args.inputs}", f"--outs={args.outputs}",
             "-f", args.formula, "--hide-status", "--dot"],
            check=True, stdout=f_dot
        )

    with open(args.hoa, "w") as f_hoa:
        subprocess.run(
            ["ltlsynt", f"--ins={args.inputs}", f"--outs={args.outputs}",
             "-f", args.formula, "--hide-status", "--hoa"],
            check=True, stdout=f_hoa
        )

    print(f"Wrote {args.dot} and {args.hoa}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import subprocess

def main():
    p = argparse.ArgumentParser(description="Generate controller.dot and controller.hoa via ltlsynt")
    p.add_argument("--inputs", required=True, help="Comma-separated input APs, e.g. a,b")
    p.add_argument("--outputs", required=True, help="Comma-separated output APs, e.g. p0,p1")
    p.add_argument("--formula", required=True, help="LTL formula (quote it!)")
    p.add_argument("--dot", default="controller.dot", help="Output DOT path (default: controller.dot)")
    p.add_argument("--hoa", default="controller.hoa", help="Output HOA path (default: controller.hoa)")
    args = p.parse_args()

    # Run ltlsynt once per format, capturing stdout directly to files (no shell redirection)
    with open(args.dot, "w") as f_dot:
        subprocess.run(
            ["ltlsynt", f"--ins={args.inputs}", f"--outs={args.outputs}",
             "-f", args.formula, "--hide-status", "--dot"],
            check=True, stdout=f_dot
        )

    with open(args.hoa, "w") as f_hoa:
        subprocess.run(
            ["ltlsynt", f"--ins={args.inputs}", f"--outs={args.outputs}",
             "-f", args.formula, "--hide-status", "--hoa"],
            check=True, stdout=f_hoa
        )

    print(f"Wrote {args.dot} and {args.hoa}")

if __name__ == "__main__":
    main()
