from plsatwitter.utils.unix import get_username

root_folder = "/home/{}/plsatwitter-results/".format(get_username())


folders = {
        "svd": root_folder + "svd/",
        "topics": root_folder + "topics/",
        "csv": root_folder + "csv/",
        "frobenius": root_folder + "frobenius/",
        "divergencia": root_folder + "divergencia/"
}
