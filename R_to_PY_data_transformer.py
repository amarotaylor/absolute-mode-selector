"""Usage: R_to_PY_data_transformer.py [-p P] [-o O]

Options:
    -p P    The path to R data files [default: .]
    -o O    The desired path to output the CSV files [default: ./output]
"""

import os
import docopt
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

col_names = ['alpha', 'tau', 'AT', 'b', 'delta', 'LL', 'mode_curv', 'genome mass', 'sigma.h.hat', 'theta.z.hat', 'sigma.A.hat',
             'theta.Q.hat', 'lambda.hat', 'theta.0', 'frac.het', 'SCNA_LL', 'entropy', 'Kar_LL', 'WGD', 'combined_LL',
             'SSNV_LL', 'SCNA_Theta_integral', 'dens']

# path to R data files
in_path = 'C:/Users/Adam Yaari/PycharmProjects/AmarosProject/R_data'

# path of desired output
out_path = None

# Build R function to be used as a python package
load_RData_func_str = """
   load_RData <- function(file_path) {
      load(file_path)
      head_name <- ls()[1]
      file_name <- names(`segobj.list`)[1]
      r_data <- `segobj.list`[[file_name]]$mode.res$mode.tab
      return(r_data)
  }
  """


def convertRtoPandas(file_path):
    # Pack the function above as a package
    r_pack = SignatureTranslatedAnonymousPackage(load_RData_func_str, "r_pack")

    pandas2ri.activate()
    r_data = r_pack.load_RData(file_path)
    py_data = pd.DataFrame(pandas2ri.ri2py(r_data), columns=col_names)
    pandas2ri.deactivate()

    return py_data


if __name__ == "__main__":
    # Assign user input arguments
    arguments = docopt.docopt(__doc__)
    in_path = arguments['-p'].strip()
    out_path = arguments['-o'].strip()

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for file_name in os.listdir(in_path):
        if file_name.endswith('.RData'):
            convertRtoPandas(in_path + "/" + file_name).to_csv(out_path + "/" + file_name.split('.')[0] + ".csv")