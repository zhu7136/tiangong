import datetime
import io
import tempfile
from pathlib import Path
import tarfile
import unittest
from private_bootstrap import validate_config, unpack

class Tests(unittest.TestCase):
    def config(self):
        return dict(schema_version=1,enabled=True,mode='preflight',archive_bytes=101291988,
                    archive_sha256='5ad9383f74968fae4740536674ccb049c37746fb656acb116413767cd60135d0',
                    start_before_utc_epoch=1000,
                    archive_url='https://storage.googleapis.com/test?X-Goog-Date=19700101T000000Z&X-Goog-Expires=86400')
    def test_bounded_config(self):
        self.assertEqual(validate_config(self.config(),now=100),86400)
    def test_queue_delay(self):
        with self.assertRaises(ValueError):validate_config(self.config(),now=1001)
    def test_reject_expiry_after_wait(self):
        c=self.config();c['archive_url']=c['archive_url'].replace('86400','3000')
        with self.assertRaises(ValueError):validate_config(c,now=100)
    def test_reject_wrong_host(self):
        c=self.config();c['archive_url']=c['archive_url'].replace('storage.googleapis.com','invalid.example')
        with self.assertRaises(ValueError):validate_config(c,now=100)
    def test_reject_extra_credentials(self):
        c=self.config();c['api_key']='not-a-real-key'
        with self.assertRaises(ValueError):validate_config(c,now=100)
    def test_reject_disabled(self):
        c=self.config();c['enabled']=False
        with self.assertRaises(ValueError):validate_config(c,now=100)
    def test_reject_traversal(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            archive=Path(d)/'bad.tar.gz'
            with tarfile.open(archive,'w:gz') as f:
                for i in range(134):
                    name='../escape' if i==0 else str(i)
                    info=tarfile.TarInfo(name);info.size=1
                    f.addfile(info,io.BytesIO(b'x'))
            with self.assertRaises(ValueError):unpack(archive,Path(d)/'out')

if __name__=='__main__':unittest.main()