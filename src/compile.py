#! /usr/bin/env python3
import os
import argparse

def execute_nvidia_command(filename, args):
    basename = os.path.splitext(filename)[0]
    if args.warn_as_error:
        warning_code = "-Xcompiler -Wall -Werror all-warnings -Xcompiler"
    else:
        warning_code = "-Xcompiler -Wall -Xcompiler"
    arch_code = "-gencode arch=compute_75,code=sm_75"
    optim_code = "-O3"
    include_dir = ""
    if os.getenv("CUTLASS_PATH"):
        include_dir = "-I" + os.getenv("CUTLASS_PATH") + "/include"
    if args.compile:
        output_command = f"-c -o {basename}.o {filename}"
    else:
        output_command = f"-o {basename} {filename}"
    cmd = f"nvcc {warning_code} -g -G -std=c++20 {include_dir} {optim_code} {arch_code} {output_command}"
    if os.system(cmd):
        print(f'[Error] {cmd}')
        return
    else:
        print(f'[Success] {cmd}')
    cmd = f"nvcc {warning_code} -g -G -std=c++20 {include_dir} {optim_code} {arch_code} -ptx -o {basename}.ptx {filename}"
    if os.system(cmd):
        print(f'[Error] {cmd}')
        return
    else:
        print(f'[Success] {cmd}')
    if args.compile:
        cmd = f"cuobjdump -ptx -sass {basename}.o > {basename}.sass"
    else:
        cmd = f"cuobjdump -ptx -sass {basename} > {basename}.sass"
    if os.system(cmd):
        print(f'[Error] {cmd}')
        return
    else:
        print(f'[Success] {cmd}')
    
    # sudo visudo /usr/local/cuda/bin/
    if args.nsys:
        nsys_trace = "--trace=cuda,nvtx,cublas,osrt,cudnn"
        cmd = f"nsys profile {nsys_trace} --force-overwrite true -o {basename}.nsys-out {basename}"
        if os.system(cmd):
            print(f'[Error] {cmd}')
            return
        else:
            print(f'[Success] {cmd}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warn-as-error", action="store_true")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.filename):
        print(f"{args.filename} does not exist")
        exit(1)
    execute_nvidia_command(args.filename, args)
    

if __name__ == "__main__":
    main()

