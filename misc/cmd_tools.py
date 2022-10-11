import subprocess
import shlex


def run_cmd(shell_cmd, shell=False):
    # shell_cmd = 'python run_ssl_downtask.py' # 假如cut_words有参数
    cmd = shlex.split(shell_cmd)
    print(cmd)
    p = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Check if child process has terminated. Set and return returncode
    out_lines = []
    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip()
        if line:
            out_lines.append(str(line))
            print('Subprogram output: [{}]'.format(line))
    if p.returncode == 0:
        print('Subprogram success')
    else:
        print('Subprogram failed')
    return out_lines


def cmd_success(shell_cmd, shell=False):
    # shell_cmd = 'python run_ssl_downtask.py' # 假如cut_words有参数
    cmd = shlex.split(shell_cmd)
    # print(cmd)
    p = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Check if child process has terminated. Set and return returncode
    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip()
        # if line:
        #     print('Subprogram output: [{}]'.format(line))
    if p.returncode == 0:
        return True
    else:
        return False