import tempfile

from jack.util.conf import load_config


def test_load_config():
    with tempfile.TemporaryDirectory() as tmp_dir:
        conf_1 = """
            seed: 1
            batch_size: 32
        """
        conf_2 = """
            batch_size: 64
            parent_config: {parent_config}
        """.format(parent_config=tmp_dir + "/conf1.yaml")

        with open(tmp_dir + "/conf1.yaml", "w") as file_1:
            file_1.write(conf_1)
        with open(tmp_dir + "/conf2.yaml", "w") as file_2:
            file_2.write(conf_2)

        conf = load_config(tmp_dir + "/conf2.yaml")
        print(conf)
        assert conf['batch_size'] == 64
        assert conf['seed'] == 1
        assert conf['parent_config'] == tmp_dir + "/conf1.yaml"
