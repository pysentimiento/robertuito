import os
import fire
import asyncio
import aiofiles
import glob
import re
from finetune_vs_scratch.preprocessing import special_tokens
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer



async def worker(name, queue, pbar, out_dir, tokenizer=None):
    """
    Tokenize tweets and add them if they are longer than six tokens
    """
    file_path = os.path.join(out_dir, f"{name}.txt")

    async with aiofiles.open(file_path, "w+") as f:
        while True:
            # Get a "work item" out of the queue.
            batch = await queue.get()

            # Count emojis just once
            to_be_tokenized_batch = [re.sub("emoji.*?emoji", "emoji", tw) for tw in batch]
            tokenized_batch = tokenizer(to_be_tokenized_batch, truncation=True)
            lens = [sum(x)-2 for x in tokenized_batch["attention_mask"]]
            # Los que vamos a salvar
            tweets = [tweet for (tweet, l) in zip(batch, lens) if l >= 6]

            await f.write("\n".join(tweets) + "\n")

            # Notify the queue that the "work item" has been processed.
            pbar.update(len(batch))
            queue.task_done()


def process_file(file_path, queue, batch_size=4096):
    """
    Process file and add it to asyncio queue
    """
    with open(file_path, "r") as f:
        tweets = []
        for tweet in f:
            tweets.append(tweet.rstrip("\n"))
            if len(tweets) >= batch_size:
                queue.put_nowait(tweets)
                tweets = []

        if tweets:
            queue.put_nowait(tweets)

async def main(in_dir, out_dir, num_workers):
    """
    Event loop
    """
    # Contado ccon wc -l
    total_tweets = 618654625
    pbar = tqdm(total=total_tweets)

    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    tokenizer.model_max_length = 128
    tokenizer.add_tokens(special_tokens)


    in_files = glob.glob(os.path.join(in_dir, "*.txt"))
    queue = asyncio.Queue()

    print("Procesando tweets")
    # Create three worker tasks to process the queue concurrently.
    pbar = tqdm(total=total_tweets)
    print(f"Creando {num_workers} workers")
    tasks = []
    for i in range(num_workers):
        task = asyncio.create_task(worker(f'spanish-tweets-{str(i).zfill(3)}', queue, pbar, out_dir, tokenizer))
        tasks.append(task)

    print("Comenzando!")

    filepbar = tqdm(total=len(in_files))
    for file_path in in_files:
        process_file(file_path, queue)
        await queue.join()
        filepbar.update()


    for task in tasks:
        task.cancel()
    # Wait until all worker tasks are cancelled.
    await asyncio.gather(*tasks, return_exceptions=True)
    return


def filter_tweets(in_dir, out_dir, num_workers):
    """
    Preprocess files

    """

    asyncio.run(main(in_dir, out_dir, num_workers))



if __name__ == "__main__":
    fire.Fire(filter_tweets)
