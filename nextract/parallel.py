"""
Parallel processing utilities for Nextract.

Provides ThreadPoolExecutor-based parallel processing with comprehensive
error handling, logging, and progress tracking.
"""

from __future__ import annotations

import concurrent.futures
import traceback
from typing import Callable, TypeVar, Any
from dataclasses import dataclass
import structlog

log = structlog.get_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class ProcessingError:
    """Error information for failed processing"""
    item_index: int
    error_type: str
    error_message: str
    traceback: str
    item_repr: str


@dataclass
class BatchResult:
    """Result of batch processing"""
    results: list[Any]
    errors: list[ProcessingError]
    successful_count: int
    failed_count: int
    total_count: int


class ParallelProcessingError(Exception):
    """Raised when parallel processing encounters errors"""
    
    def __init__(self, message: str, errors: list[ProcessingError]):
        super().__init__(message)
        self.errors = errors
    
    def __str__(self):
        error_summary = "\n".join([
            f"  [{i+1}] Item {err.item_index}: {err.error_type}: {err.error_message}"
            for i, err in enumerate(self.errors[:5])  # Show first 5 errors
        ])
        
        if len(self.errors) > 5:
            error_summary += f"\n  ... and {len(self.errors) - 5} more errors"
        
        return f"{super().__str__()}\n\nErrors:\n{error_summary}"


class ParallelProcessor:
    """
    Process items in parallel using ThreadPoolExecutor.
    
    Features:
    - Configurable worker pool size
    - Batch processing for memory efficiency
    - Comprehensive error handling and logging
    - Progress tracking
    - Maintains result order
    
    Example:
        processor = ParallelProcessor(max_workers=10)
        results = processor.process_batch(
            items=[1, 2, 3, 4, 5],
            process_fn=lambda x: x * 2
        )
    """
    
    def __init__(self, max_workers: int = 10):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of parallel workers (default: 10)
        
        Raises:
            ValueError: If max_workers < 1
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        
        self.max_workers = max_workers
        log.info("parallel_processor_initialized", max_workers=max_workers)
    
    def process_batch(
        self,
        items: list[T],
        process_fn: Callable[[T], R],
        batch_size: int | None = None,
        fail_fast: bool = False,
        return_errors: bool = False
    ) -> list[R] | BatchResult:
        """
        Process items in parallel batches.
        
        Args:
            items: List of items to process
            process_fn: Function to apply to each item
            batch_size: Number of items per batch (default: process all at once)
            fail_fast: If True, raise exception on first error (default: False)
            return_errors: If True, return BatchResult with errors (default: False)
        
        Returns:
            List of results in original order (if return_errors=False)
            BatchResult with results and errors (if return_errors=True)
        
        Raises:
            ParallelProcessingError: If fail_fast=True and any item fails
            ValueError: If items is empty
        
        Example:
            # Simple usage
            results = processor.process_batch(
                items=[1, 2, 3],
                process_fn=lambda x: x * 2
            )
            # results = [2, 4, 6]
            
            # With error handling
            batch_result = processor.process_batch(
                items=[1, 2, 3],
                process_fn=lambda x: x * 2,
                return_errors=True
            )
            # batch_result.results = [2, 4, 6]
            # batch_result.errors = []
        """
        if not items:
            return BatchResult(results=[], errors=[], total_count=0, successful_count=0, failed_count=0)
        
        log.info(
            "parallel_processing_started",
            total_items=len(items),
            batch_size=batch_size or len(items),
            max_workers=self.max_workers,
            fail_fast=fail_fast
        )
        
        # If no batch_size specified, process all at once
        if batch_size is None:
            batch_size = len(items)
        
        all_results = []
        all_errors = []
        
        # Process in batches
        for batch_num, i in enumerate(range(0, len(items), batch_size), start=1):
            batch = items[i:i+batch_size]
            
            log.info(
                "processing_batch",
                batch_num=batch_num,
                batch_start=i,
                batch_size=len(batch),
                workers=min(self.max_workers, len(batch))
            )
            
            try:
                batch_results, batch_errors = self._process_single_batch(
                    batch=batch,
                    process_fn=process_fn,
                    batch_start_idx=i,
                    fail_fast=fail_fast
                )
                
                all_results.extend(batch_results)
                all_errors.extend(batch_errors)
                
                log.info(
                    "batch_completed",
                    batch_num=batch_num,
                    successful=len(batch_results) - len(batch_errors),
                    failed=len(batch_errors)
                )
                
            except ParallelProcessingError as e:
                # If fail_fast=True, propagate immediately
                log.error(
                    "batch_failed_fast",
                    batch_num=batch_num,
                    error_count=len(e.errors)
                )
                raise
        
        # Log final summary
        successful_count = sum(1 for r in all_results if r is not None)
        log.info(
            "parallel_processing_completed",
            total_items=len(items),
            successful=successful_count,
            failed=len(all_errors),
            success_rate=f"{successful_count/len(items)*100:.1f}%"
        )

        # Return based on return_errors flag
        if return_errors:
            return BatchResult(
                results=all_results,
                errors=all_errors,
                successful_count=successful_count,
                failed_count=len(all_errors),
                total_count=len(items)
            )
        else:
            # If there were errors and not fail_fast, log warning
            if all_errors:
                log.warning(
                    "processing_completed_with_errors",
                    error_count=len(all_errors),
                    message="Some items failed processing. Use return_errors=True to get error details."
                )
            return [r for r in all_results if r is not None]
    
    def _process_single_batch(
        self,
        batch: list[T],
        process_fn: Callable[[T], R],
        batch_start_idx: int,
        fail_fast: bool
    ) -> tuple[list[R], list[ProcessingError]]:
        """
        Process a single batch in parallel.
        
        Returns:
            Tuple of (results, errors)
            - results: List with results or None for failed items
            - errors: List of ProcessingError for failed items
        """
        results = [None] * len(batch)
        errors = []
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(batch))
        ) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._safe_process, process_fn, item, batch_start_idx + idx): idx
                for idx, item in enumerate(batch)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                global_idx = batch_start_idx + idx
                
                try:
                    result, error = future.result()
                    
                    if error:
                        # Processing failed
                        errors.append(error)
                        results[idx] = None
                        
                        log.error(
                            "item_processing_failed",
                            item_index=global_idx,
                            error_type=error.error_type,
                            error_message=error.error_message
                        )
                        
                        # If fail_fast, raise immediately
                        if fail_fast:
                            raise ParallelProcessingError(
                                f"Processing failed for item {global_idx}: {error.error_message}",
                                errors=[error]
                            )
                    else:
                        # Processing succeeded
                        results[idx] = result
                        log.debug("item_processing_succeeded", item_index=global_idx)
                
                except Exception as e:
                    # This should not happen (future.result() should not raise)
                    # but handle it just in case
                    error = ProcessingError(
                        item_index=global_idx,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        traceback=traceback.format_exc(),
                        item_repr="<unknown>"
                    )
                    errors.append(error)
                    results[idx] = None
                    
                    log.error(
                        "unexpected_future_error",
                        item_index=global_idx,
                        error=str(e),
                        traceback=traceback.format_exc()
                    )
                    
                    if fail_fast:
                        raise ParallelProcessingError(
                            f"Unexpected error processing item {global_idx}: {str(e)}",
                            errors=[error]
                        )
        
        return results, errors
    
    def _safe_process(
        self,
        process_fn: Callable[[T], R],
        item: T,
        item_index: int
    ) -> tuple[R | None, ProcessingError | None]:
        """
        Safely process a single item, catching all exceptions.
        
        Returns:
            Tuple of (result, error)
            - If successful: (result, None)
            - If failed: (None, ProcessingError)
        """
        try:
            result = process_fn(item)
            return (result, None)
        
        except Exception as e:
            # Capture full error information
            error = ProcessingError(
                item_index=item_index,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                item_repr=repr(item)[:200]  # Limit repr length
            )
            return (None, error)

