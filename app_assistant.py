
import os, sys
from streamlit.web import cli as stcli
import subprocess
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(sys.argv[0]))
    print(here)
    subprocess.run(f"cd {here}", shell=True)
    script = os.path.join(here, "assistant_0.2.py")
    sys.argv = ["streamlit", "run", script, "--server.headless=true"]
    sys.exit(stcli.main())