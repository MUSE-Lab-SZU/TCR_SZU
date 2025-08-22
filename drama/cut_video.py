from traceback import format_exc
from multiprocessing import cpu_count
from argparse import ArgumentParser

import euler
from euler.base_compat_middleware import server_middleware
euler.install_thrift_import_hook()

from atom.utils.logger import logger
from atom.video.video_edit_vcloud import cloud_video_concat
from atom.apps.micro_dramas.mago_drama_smart_cut import MagoSmartDramaCut
from atom.modules.drama_storyline_creative_generation.utils.cutting_utils import transfer_to_mago_script

from atom.idls.drama_cut.drama_video_cut_thrift import (
    DramaCutRequest,
    DramaCutResponse,
    DramaCutService,
)

server = euler.Server(DramaCutService)
server.use(server_middleware)

@server.register('cut_drama')
def cut_drama(ctx, req: DramaCutRequest) -> DramaCutResponse:
    logger.info(f"Received request: {req}")
    book_id = req.book_id
    source = req.source
    cut_script = eval(req.cut_script)
    try:
        cutter = MagoSmartDramaCut()
        mago_script = transfer_to_mago_script(book_id, source, cut_script)
        concat_segs, _ = cutter.fix_cut_seg_with_asr(mago_script, dedup_shots=True, compute_cover_ratio=False)
        concat_vid_segs = cutter.get_concat_vid_segs(concat_segs)
        vid = cloud_video_concat(concat_vid_segs)
        msg = "success"
    except Exception as e:
        logger.info(f"剪辑失败：\n{format_exc()}")
        vid = "failed"
        msg = str(e)
    finally:
        return DramaCutResponse(vid=vid, msg=msg)

if __name__ == '__main__':
    parser = ArgumentParser("Upload Review Service")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=cpu_count())
    args = parser.parse_args()
    logger.info("Starting server...")
    try:
        server.run('tcp://[::]:{}'.format(args.port), workers_count=args.workers)
    except:
        logger.error(format_exc())
    finally:
        server.stop()
        logger.info("Server exiting")
        